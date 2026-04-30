"""Loaders for the three canonical evaluation prompts.

Per spec §13 hard rules, the same prompt text must be used across every
contender. These loaders are the single source of truth — adapters call them
during ``analyze`` and pass the resulting string through unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PROMPTS_DIR = Path(__file__).resolve().parent


def load_generic() -> str:
    """Return the canonical Prompt A (Generic Brief) text."""
    return (_PROMPTS_DIR / "generic.txt").read_text(encoding="utf-8").strip()


def load_targeted(query: str) -> str:
    """Return Prompt B with the per-clip targeted query injected."""
    template = (_PROMPTS_DIR / "targeted_template.txt").read_text(encoding="utf-8").strip()
    return template.format(query=query)


def load_hallucination_trap(query: str) -> str:
    """Return Prompt C with the per-clip hallucination trap injected."""
    template = (
        (_PROMPTS_DIR / "hallucination_trap_template.txt").read_text(encoding="utf-8").strip()
    )
    return template.format(query=query)


def load_judge_rubric(
    *,
    ground_truth_summary: str,
    ground_truth_objects: list[str],
    ground_truth_ocr: list[str],
    expected_temporal_events: list[dict[str, Any]],
    targeted_query: str,
    hallucination_trap: str,
    generic_response: str,
    targeted_response: str,
    hallucination_trap_response: str,
    ocr_applicable: str,
) -> str:
    """Return the judge rubric prompt with per-call ground truth and model responses injected."""
    template = (_PROMPTS_DIR / "judge_rubric_template.txt").read_text(encoding="utf-8").strip()
    return template.format(
        ground_truth_summary=ground_truth_summary,
        ground_truth_objects=json.dumps(ground_truth_objects),
        ground_truth_ocr=json.dumps(ground_truth_ocr),
        expected_temporal_events=json.dumps(expected_temporal_events),
        targeted_query=targeted_query,
        hallucination_trap=hallucination_trap,
        generic_response=generic_response,
        targeted_response=targeted_response,
        hallucination_trap_response=hallucination_trap_response,
        ocr_applicable=ocr_applicable,
    )
