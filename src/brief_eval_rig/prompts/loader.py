"""Loaders for the three canonical evaluation prompts.

Per spec §13 hard rules, the same prompt text must be used across every
contender. These loaders are the single source of truth — adapters call them
during ``analyze`` and pass the resulting string through unchanged.
"""

from __future__ import annotations

from pathlib import Path

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
