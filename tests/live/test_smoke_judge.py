"""Live smoke for the LLM judge — opt-in via `pytest -m live`."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from brief_eval_rig.corpus.loader import load_clip
from brief_eval_rig.scoring.llm_judge import LLMJudge

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_JSON = REPO_ROOT / "corpus" / "sample" / "sample-001.json"
OUTPUTS_DIR = REPO_ROOT / "outputs" / "smp-001"


def _load_output_or_skip(name: str) -> dict[str, object]:
    path = OUTPUTS_DIR / name
    if not path.is_file():
        pytest.skip(f"target output not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts") or {}
    for kind in ("generic", "targeted", "hallucination_trap"):
        err = (prompts.get(kind) or {}).get("error")
        if err:
            pytest.skip(f"{name} has upstream error on {kind!r}: {err}")
    assert isinstance(payload, dict)
    return payload


@pytest.mark.live
def test_judge_smoke_anthropic_route(tmp_path: Path) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set; skipping Anthropic-route smoke")
    if not SAMPLE_JSON.is_file():
        pytest.skip(f"corpus sample missing: {SAMPLE_JSON}")

    model_output = _load_output_or_skip("qwen3-6-flash.json")
    clip = load_clip(SAMPLE_JSON)

    judge = LLMJudge()
    run, dimension_scores = judge.judge_pair(
        clip_metadata=clip,
        model_output=model_output,
        lineup_min_latency_ms=1000,
        lineup_min_paid_cost_usd=0.001,
    )

    assert run.error is None, f"unexpected judge error: {run.error}"
    assert run.composite is not None
    assert 0 <= run.composite <= 10
    assert run.judge_model == "claude-opus-4-7"
    assert len(dimension_scores) == 7

    assert run.raw_judge_response is not None
    raw_obj = json.loads(run.raw_judge_response)
    usage = raw_obj.get("usage") or {}
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    assert cache_write > 0 or cache_read > 0, (
        "expected Anthropic prompt cache to be engaged "
        f"(cache_creation_input_tokens={cache_write}, "
        f"cache_read_input_tokens={cache_read})"
    )


@pytest.mark.live
def test_judge_smoke_openai_route(tmp_path: Path) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping OpenAI-route smoke")
    if not SAMPLE_JSON.is_file():
        pytest.skip(f"corpus sample missing: {SAMPLE_JSON}")

    model_output = _load_output_or_skip("claude-sonnet-4-6.json")
    clip = load_clip(SAMPLE_JSON)

    judge = LLMJudge()
    run, dimension_scores = judge.judge_pair(
        clip_metadata=clip,
        model_output=model_output,
        lineup_min_latency_ms=1000,
        lineup_min_paid_cost_usd=0.001,
    )

    assert run.error is None, f"unexpected judge error: {run.error}"
    assert run.composite is not None
    assert 0 <= run.composite <= 10
    assert run.judge_model == "gpt-5"
    assert len(dimension_scores) == 7
