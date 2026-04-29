"""Ollama adapter (frame-samples; Ollama API supports images only, no native video).

One adapter class configurable for the three local contenders shipping in Phase 2:

- Qwen3.6 27B (``qwen3.6:27b``)
- Nemotron 3 Nano Omni (``nemotron3``, 33B, reasoning model)
- Gemma 4 26B A4B (``gemma4:26b``)

Per Ollama docs (verified 2026-04-29), ``/api/chat`` accepts an ``images`` array
of base64 strings; there is no dedicated video field. All three adapters
frame-sample using Phase 1's :func:`brief_eval_rig.runner.frame_sampler.sample_frames`
and send each frame as a separate image.

The same adapter works against local Ollama (default ``localhost:11434``) or
remote Ollama on Vast.ai by overriding ``base_url`` (constructor arg) or
exporting ``OLLAMA_BASE_URL``. Ollama does not require authentication; no API
key handling.

Costs are always 0.0 — local inference is free, and Vast.ai compute is
amortized at the operational layer rather than per-call.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, cast

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from brief_eval_rig.adapters.base import AnalysisResult, Clip, Deployment, Lineage, VLMAdapter
from brief_eval_rig.runner.frame_sampler import default_n_for_duration, sample_frames

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_TIMEOUT_S = 600.0


class OllamaAdapter(VLMAdapter):
    """Ollama chat-completions adapter, configurable per local model."""

    def __init__(
        self,
        *,
        model_id: str,
        lineup_name: str,
        lineage: Lineage,
        base_url: str | None = None,
        max_output_tokens: int = 1024,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._model_id = model_id
        self._lineup_name = lineup_name
        self._lineage: Lineage = lineage
        self._base_url = (
            base_url or os.environ.get("OLLAMA_BASE_URL") or _DEFAULT_BASE_URL
        ).rstrip("/")
        self._max_output_tokens = max_output_tokens
        self._timeout_s = timeout_s

    def name(self) -> str:
        return self._lineup_name

    def lineage(self) -> Lineage:
        return self._lineage

    def deployment(self) -> Deployment:
        return "local"

    def supports_native_video(self) -> bool:
        return False

    def estimate_cost(self, clip: Clip) -> float:
        return 0.0

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
        images_b64 = [base64.b64encode(f).decode("ascii") for f in frames]

        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": images_b64,
                }
            ],
            "stream": False,
            "options": {"num_predict": self._max_output_tokens},
        }

        response_json = self._post_with_retry(payload)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        message = response_json.get("message") or {}
        summary = (message.get("content") or "").strip()
        if not summary:
            # Reasoning models (Nemotron) may emit chain-of-thought into a
            # `thinking` field and only finalize content after reasoning. If we
            # ran out of tokens mid-reasoning, fall back so the run still has a
            # usable summary — same pattern as the cloud OpenRouter adapter.
            summary = (message.get("thinking") or "").strip()

        return AnalysisResult(
            summary=summary,
            raw_response=response_json,
            latency_ms=elapsed_ms,
            cost_usd=0.0,
            error=None,
        )

    @retry(
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout_s) as client:
            r = client.post(f"{self._base_url}/api/chat", json=payload)
            r.raise_for_status()
            return cast("dict[str, Any]", r.json())
