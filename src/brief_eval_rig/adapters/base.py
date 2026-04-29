"""Base abstract class and value types for VLM adapter implementations.

Concrete adapter subclasses live in Phase 1 (cloud: Claude, Gemini Pro, Gemini Flash,
Qwen Flash, Nemotron cloud) and Phase 2 (Ollama-based local: Qwen 27B, Qwen 9B,
Nemotron, Gemma 4). This module is intentionally import-stable: it pulls in no
model SDK at import time so the foundation can be tested without network access.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

Lineage = Literal["qwen", "nvidia", "google", "anthropic"]
Deployment = Literal["cloud", "local"]


class Clip(BaseModel):
    """A single video clip queued for evaluation.

    The adapter is responsible for opening ``video_path`` itself; this struct
    only carries the path and metadata. ``metadata`` is the deserialized sidecar
    JSON for the clip, validated separately by ``brief_eval_rig.corpus.schema``.
    """

    model_config = ConfigDict(frozen=True)

    clip_id: str
    video_path: Path
    duration_seconds: float
    metadata: dict[str, Any]


class AnalysisResult(BaseModel):
    """Result of running one prompt against one clip via one adapter.

    Adapters return an ``AnalysisResult`` even on failure: set ``error`` to a
    short message and leave ``summary`` empty. ``latency_ms`` and ``cost_usd``
    are populated on success; on failure they may be 0.
    """

    summary: str
    raw_response: dict[str, Any]
    latency_ms: int
    cost_usd: float
    error: str | None = None


class VLMAdapter(ABC):
    """Unified interface across all nine contender VLMs.

    Phase 1 implements the cloud adapters. Phase 2 implements the Ollama
    adapter, which serves as a single concrete class with four configurations
    (Qwen 27B local, Qwen 9B local, Nemotron local, Gemma 4 local).

    Implementations must be safe to instantiate without network access at
    construction time; lazy network calls happen inside :meth:`analyze`.
    """

    @abstractmethod
    def name(self) -> str:
        """Stable identifier used in output paths and the leaderboard.

        Format suggestion: lowercase, hyphenated, includes deployment hint
        when ambiguous (e.g., ``"qwen3.6-27b-cloud"`` vs ``"qwen3.6-27b-local"``).
        Must be unique across the lineup.
        """

    @abstractmethod
    def lineage(self) -> Lineage:
        """Model family this adapter belongs to.

        Used by the LLM-judge orchestrator to enforce the self-grading
        rule: a judge must not score outputs from its own lineage.
        """

    @abstractmethod
    def deployment(self) -> Deployment:
        """Whether this adapter calls a cloud endpoint or local runtime.

        Used by the runner for parallelism strategy (cloud calls run
        concurrently via asyncio; local calls serialize on the GPU).
        """

    @abstractmethod
    def analyze(self, clip: Clip, prompt: str) -> AnalysisResult:
        """Run the model on a single clip against a single prompt.

        Implementations must populate ``latency_ms`` (wall-clock from request
        send to response complete) and ``cost_usd`` (provider token-priced cost
        for cloud, ``0.0`` for local). Surface errors via
        :attr:`AnalysisResult.error` rather than raising, so a single failure
        does not abort the run.
        """

    @abstractmethod
    def supports_native_video(self) -> bool:
        """If ``False``, the adapter must internally frame-sample.

        The frame sampler utility ships in Phase 1. Adapters returning
        ``False`` are expected to call it during :meth:`analyze`. Adapters
        returning ``True`` pass the raw video file/URL to the provider's
        video API.
        """

    @abstractmethod
    def estimate_cost(self, clip: Clip) -> float:
        """Pre-flight cost estimate in USD for budgeting.

        Used by the runner to refuse a full eval run that would exceed the
        configured budget cap. Local adapters return ``0.0``.
        """
