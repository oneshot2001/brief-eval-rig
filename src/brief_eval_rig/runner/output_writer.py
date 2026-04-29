"""Per-clip per-model JSON output writer matching spec §9 exactly.

Phase 1 populates the prompts.{generic,targeted,hallucination_trap} fields
with real model output. Score-related fields are nulled out for Phase 3 to
fill in. Pretty-printed for diff-friendliness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from brief_eval_rig.adapters.base import AnalysisResult, VLMAdapter

PROMPT_KINDS = ("generic", "targeted", "hallucination_trap")


def _result_to_dict(result: AnalysisResult) -> dict[str, Any]:
    return {
        "response": result.summary,
        "latency_ms": result.latency_ms,
        "cost_usd": result.cost_usd,
        "error": result.error,
    }


def write_result(
    output_root: Path,
    clip_id: str,
    adapter: VLMAdapter,
    prompt_results: dict[str, AnalysisResult],
) -> Path:
    """Write outputs/{clip_id}/{adapter.name()}.json per spec §9.

    prompt_results keys must include "generic", "targeted", and
    "hallucination_trap". Returns the path written.
    """
    missing = [k for k in PROMPT_KINDS if k not in prompt_results]
    if missing:
        raise ValueError(f"prompt_results missing required keys: {missing}")

    payload: dict[str, Any] = {
        "clip_id": clip_id,
        "model": adapter.name(),
        "lineage": adapter.lineage(),
        "deployment": adapter.deployment(),
        "prompts": {kind: _result_to_dict(prompt_results[kind]) for kind in PROMPT_KINDS},
        "scores": None,
        "composite": None,
        "judge": None,
        "judge_justifications": None,
        "human_spot_checked": False,
        "human_spot_check_scores": None,
        "spot_check_disagreements": [],
    }

    out_dir = output_root / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{adapter.name()}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path
