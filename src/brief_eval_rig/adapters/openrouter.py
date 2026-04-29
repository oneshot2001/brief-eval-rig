"""OpenRouter adapter (OpenAI-compatible chat completions).

One adapter class configurable for Qwen3.6 Flash and Nemotron 3 Nano Omni
(``:free`` tier). Both models advertise native video support
(modality: ``text+image+video->text``); the exact wire format for video is
under-documented at OpenRouter, so this adapter sends the clip as a
data URI in an ``image_url`` content block — the broadly-supported pattern
across OpenAI-compatible endpoints. If a model rejects this shape we will
add a frame-sampled fallback in a follow-up.

Costs are computed from response.usage.prompt_tokens and completion_tokens
times the configured pricing constants.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, cast

import httpx
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from brief_eval_rig.adapters.base import AnalysisResult, Clip, Deployment, Lineage, VLMAdapter
from brief_eval_rig.adapters.pricing import TokenPrice

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_TIMEOUT_S = 180.0


class OpenRouterAdapter(VLMAdapter):
    """OpenRouter chat-completions adapter, configurable per model."""

    def __init__(
        self,
        *,
        model_id: str,
        lineup_name: str,
        lineage: Lineage,
        pricing: TokenPrice,
        supports_native_video: bool = True,
        max_output_tokens: int = 1024,
        api_key: str | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        load_dotenv()
        self._model_id = model_id
        self._lineup_name = lineup_name
        self._lineage: Lineage = lineage
        self._pricing = pricing
        self._native_video = supports_native_video
        self._max_output_tokens = max_output_tokens
        self._timeout_s = timeout_s
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to .env or the environment.")
        self._api_key = key

    def name(self) -> str:
        return self._lineup_name

    def lineage(self) -> Lineage:
        return self._lineage

    def deployment(self) -> Deployment:
        return "cloud"

    def supports_native_video(self) -> bool:
        return self._native_video

    def estimate_cost(self, clip: Clip) -> float:
        est_input = int(clip.duration_seconds * 200) + 200
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
        video_bytes = Path(clip.video_path).read_bytes()
        video_b64 = base64.b64encode(video_bytes).decode("ascii")
        data_uri = f"data:video/mp4;base64,{video_b64}"

        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]

        payload = {
            "model": self._model_id,
            "max_tokens": self._max_output_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/oneshot2001/brief-eval-rig",
            "X-Title": "Brief Eval Rig",
        }

        response_json = self._post_with_retry(payload, headers)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        choice = (response_json.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        summary = (message.get("content") or "").strip()

        usage = response_json.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        cost = self._pricing.cost_for(input_tokens, output_tokens)

        return AnalysisResult(
            summary=summary,
            raw_response=response_json,
            latency_ms=elapsed_ms,
            cost_usd=cost,
            error=None,
        )

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _post_with_retry(self, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout_s) as client:
            r = client.post(_OPENROUTER_URL, json=payload, headers=headers)
            r.raise_for_status()
            return cast("dict[str, Any]", r.json())
