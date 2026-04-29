"""Anthropic Claude adapter (frame-samples; Claude Messages does not ingest video).

Sends N frames as base64 JPEG image content blocks alongside the prompt text.
Cost is computed from response.usage.input_tokens and output_tokens times the
Sonnet 4.6 pricing constants. Errors are surfaced via AnalysisResult.error,
never raised — per spec §7, a single failure must not abort the run.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, cast

from anthropic import Anthropic
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from brief_eval_rig.adapters.base import AnalysisResult, Clip, Deployment, Lineage, VLMAdapter
from brief_eval_rig.adapters.pricing import CLAUDE_SONNET_4_6, TokenPrice
from brief_eval_rig.runner.frame_sampler import default_n_for_duration, sample_frames


class ClaudeAdapter(VLMAdapter):
    """Anthropic Claude Sonnet 4.6 adapter."""

    def __init__(
        self,
        *,
        model_id: str = "claude-sonnet-4-6",
        lineup_name: str = "claude-sonnet-4-6",
        pricing: TokenPrice = CLAUDE_SONNET_4_6,
        max_output_tokens: int = 1024,
        api_key: str | None = None,
    ) -> None:
        load_dotenv()
        self._model_id = model_id
        self._lineup_name = lineup_name
        self._pricing = pricing
        self._max_output_tokens = max_output_tokens
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env or the environment.")
        self._client = Anthropic(api_key=key)

    def name(self) -> str:
        return self._lineup_name

    def lineage(self) -> Lineage:
        return "anthropic"

    def deployment(self) -> Deployment:
        return "cloud"

    def supports_native_video(self) -> bool:
        return False

    def estimate_cost(self, clip: Clip) -> float:
        n = default_n_for_duration(clip.duration_seconds)
        # Rough heuristic: ~1500 image tokens per 768x768 frame, plus prompt + output.
        est_input = n * 1500 + 200
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
        n_frames = default_n_for_duration(clip.duration_seconds)
        frames = sample_frames(Path(clip.video_path), n_frames=n_frames)

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for jpeg_bytes in frames:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(jpeg_bytes).decode("ascii"),
                    },
                }
            )

        response = self._call_with_retry(content)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        text_blocks = [b for b in response.content if getattr(b, "type", None) == "text"]
        summary = "".join(getattr(b, "text", "") for b in text_blocks)
        usage = response.usage
        cost = self._pricing.cost_for(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )

        return AnalysisResult(
            summary=summary,
            raw_response=cast("dict[str, Any]", response.model_dump()),
            latency_ms=elapsed_ms,
            cost_usd=cost,
            error=None,
        )

    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _call_with_retry(self, content: list[dict[str, Any]]) -> Any:
        # Anthropic's MessageParam content type is a TypedDict union; our dict
        # shape conforms at runtime but mypy can't narrow it. Cast to Any to
        # bypass the static check without sacrificing the rest of strict mode.
        messages = cast("Any", [{"role": "user", "content": content}])
        return self._client.messages.create(
            model=self._model_id,
            max_tokens=self._max_output_tokens,
            messages=messages,
        )
