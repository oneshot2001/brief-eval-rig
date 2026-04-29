"""Google Gemini adapter (native video via Files API).

One adapter class configurable for both Gemini 3.1 Pro Preview and Gemini 3
Flash Preview. Uploads the clip via the google-genai Files API, polls for
ACTIVE state, then references the file URI in the generate_content call.

Costs are computed from response.usage_metadata.prompt_token_count and
candidates_token_count times the configured pricing constants. Video tokens
bill at the same per-token rate as text/image inputs (verified 2026-04-29).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from brief_eval_rig.adapters.base import AnalysisResult, Clip, Deployment, Lineage, VLMAdapter
from brief_eval_rig.adapters.pricing import TokenPrice

_FILE_PROCESSING_TIMEOUT_S = 120
_FILE_PROCESSING_POLL_S = 2


class GoogleAdapter(VLMAdapter):
    """Configurable Gemini adapter for Gemini 3.1 Pro Preview / Gemini 3 Flash Preview."""

    def __init__(
        self,
        *,
        model_id: str,
        lineup_name: str,
        pricing: TokenPrice,
        max_output_tokens: int = 1024,
        api_key: str | None = None,
    ) -> None:
        load_dotenv()
        self._model_id = model_id
        self._lineup_name = lineup_name
        self._pricing = pricing
        self._max_output_tokens = max_output_tokens
        key = api_key or os.environ.get("GOOGLE_AI_STUDIO_API_KEY", "")
        if not key:
            raise RuntimeError(
                "GOOGLE_AI_STUDIO_API_KEY is not set. Add it to .env or the environment."
            )
        self._client = genai.Client(api_key=key)

    def name(self) -> str:
        return self._lineup_name

    def lineage(self) -> Lineage:
        return "google"

    def deployment(self) -> Deployment:
        return "cloud"

    def supports_native_video(self) -> bool:
        return True

    def estimate_cost(self, clip: Clip) -> float:
        # Crude heuristic: ~250 tokens/sec of video for Gemini, plus prompt + output.
        est_input = int(clip.duration_seconds * 250) + 200
        est_output = 800
        return self._pricing.cost_for(est_input, est_output)

    def analyze(self, clip: Clip, prompt: str) -> AnalysisResult:
        start = time.monotonic()
        try:
            return self._analyze_inner(clip, prompt, start)
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            return AnalysisResult(
                summary="",
                raw_response={},
                latency_ms=elapsed_ms,
                cost_usd=0.0,
                error=f"{type(e).__name__}: {e}",
            )

    def _analyze_inner(self, clip: Clip, prompt: str, start: float) -> AnalysisResult:
        video_path = Path(clip.video_path)
        uploaded = self._client.files.upload(file=str(video_path))
        uploaded = self._wait_for_active(uploaded)

        response = self._client.models.generate_content(
            model=self._model_id,
            contents=[uploaded, prompt],
            config=genai_types.GenerateContentConfig(
                max_output_tokens=self._max_output_tokens,
            ),
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        summary = (response.text or "").strip()

        usage = response.usage_metadata
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
        cost = self._pricing.cost_for(input_tokens, output_tokens)

        return AnalysisResult(
            summary=summary,
            raw_response=response.model_dump(),
            latency_ms=elapsed_ms,
            cost_usd=cost,
            error=None,
        )

    def _wait_for_active(self, file_obj: Any) -> Any:
        """Poll Files API until the upload is ACTIVE or fails."""
        deadline = time.monotonic() + _FILE_PROCESSING_TIMEOUT_S
        current = file_obj
        while time.monotonic() < deadline:
            state = getattr(current, "state", None)
            state_name = getattr(state, "name", str(state))
            if state_name == "ACTIVE":
                return current
            if state_name == "FAILED":
                raise RuntimeError(f"Gemini Files API processing failed for {file_obj.name}")
            time.sleep(_FILE_PROCESSING_POLL_S)
            current = self._client.files.get(name=current.name)
        raise TimeoutError(
            f"Gemini Files API did not reach ACTIVE within {_FILE_PROCESSING_TIMEOUT_S}s"
        )
