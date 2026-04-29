"""Model adapters. Concrete implementations land in Phase 1 (cloud) and Phase 2 (local)."""

from brief_eval_rig.adapters.base import (
    AnalysisResult,
    Clip,
    Deployment,
    Lineage,
    VLMAdapter,
)

__all__ = ["AnalysisResult", "Clip", "Deployment", "Lineage", "VLMAdapter"]
