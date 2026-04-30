"""Output writer tests verify spec §9 schema compliance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brief_eval_rig.adapters.base import AnalysisResult, Clip, Deployment, Lineage, VLMAdapter
from brief_eval_rig.runner.output_writer import update_with_judge, write_result
from brief_eval_rig.scoring.storage import DimensionScore, JudgeRun


class _StubAdapter(VLMAdapter):
    def name(self) -> str:
        return "stub-adapter"

    def lineage(self) -> Lineage:
        return "anthropic"

    def deployment(self) -> Deployment:
        return "cloud"

    def supports_native_video(self) -> bool:
        return False

    def estimate_cost(self, clip: Clip) -> float:
        return 0.0

    def analyze(self, clip: Clip, prompt: str) -> AnalysisResult:
        return AnalysisResult(summary="", raw_response={}, latency_ms=0, cost_usd=0.0)


def _result(text: str = "ok", cost: float = 0.01) -> AnalysisResult:
    return AnalysisResult(
        summary=text,
        raw_response={"id": "abc"},
        latency_ms=1234,
        cost_usd=cost,
        error=None,
    )


def test_write_result_creates_file(tmp_path: Path) -> None:
    adapter = _StubAdapter()
    results = {
        "generic": _result("g"),
        "targeted": _result("t"),
        "hallucination_trap": _result("h"),
    }
    out = write_result(tmp_path, "smp-001", adapter, results)
    assert out.exists()
    assert out == tmp_path / "smp-001" / "stub-adapter.json"


def test_write_result_schema_matches_spec(tmp_path: Path) -> None:
    adapter = _StubAdapter()
    results = {
        "generic": _result("g", 0.01),
        "targeted": _result("t", 0.02),
        "hallucination_trap": _result("h", 0.03),
    }
    out = write_result(tmp_path, "smp-001", adapter, results)
    data = json.loads(out.read_text())

    # Top-level shape per spec §9.
    assert data["clip_id"] == "smp-001"
    assert data["model"] == "stub-adapter"
    assert data["lineage"] == "anthropic"
    assert data["deployment"] == "cloud"
    assert set(data["prompts"].keys()) == {"generic", "targeted", "hallucination_trap"}

    for kind in ("generic", "targeted", "hallucination_trap"):
        block = data["prompts"][kind]
        assert set(block.keys()) == {"response", "latency_ms", "cost_usd", "error"}

    # Phase 3 fields are nulled out.
    assert data["scores"] is None
    assert data["composite"] is None
    assert data["judge"] is None
    assert data["judge_justifications"] is None
    assert data["human_spot_checked"] is False
    assert data["human_spot_check_scores"] is None
    assert data["spot_check_disagreements"] == []


def test_write_result_rejects_missing_prompt_kind(tmp_path: Path) -> None:
    adapter = _StubAdapter()
    incomplete = {"generic": _result(), "targeted": _result()}
    with pytest.raises(ValueError):
        write_result(tmp_path, "smp-001", adapter, incomplete)


def test_write_result_pretty_printed(tmp_path: Path) -> None:
    adapter = _StubAdapter()
    results = {
        "generic": _result(),
        "targeted": _result(),
        "hallucination_trap": _result(),
    }
    out = write_result(tmp_path, "smp-001", adapter, results)
    raw = out.read_text()
    assert "\n  " in raw  # 2-space indent present
    assert raw.endswith("\n")


_JUDGE_DIMS = (
    "description_accuracy",
    "ocr_fidelity",
    "spatial_temporal",
    "targeted_query",
    "hallucination_resistance",
    "court_grade_language",
    "speed_cost",
)


def _seed_json(tmp_path: Path) -> Path:
    adapter = _StubAdapter()
    results = {
        "generic": _result("g"),
        "targeted": _result("t"),
        "hallucination_trap": _result("h"),
    }
    return write_result(tmp_path, "smp-001", adapter, results)


def _success_run(run_id: str = "rid-1") -> tuple[JudgeRun, list[DimensionScore]]:
    run = JudgeRun(
        judge_run_id=run_id,
        clip_id="smp-001",
        model="stub-adapter",
        judge_model="gpt-5",
        composite=7.42,
        error=None,
        judge_input_tokens=1234,
        judge_output_tokens=567,
        judge_cost_usd=0.0123,
        raw_judge_response='{"ok": true}',
        created_at="2026-04-30T18:00:00Z",
    )
    scores: list[DimensionScore] = []
    for dim in _JUDGE_DIMS:
        scores.append(
            DimensionScore(
                judge_run_id=run_id,
                dimension=dim,  # type: ignore[arg-type]
                score=8 if dim != "ocr_fidelity" else None,
                justification=f"j-{dim}",
            )
        )
    return run, scores


def test_update_with_judge_round_trip(tmp_path: Path) -> None:
    out = _seed_json(tmp_path)
    pre = json.loads(out.read_text())
    run, scores = _success_run()
    update_with_judge(out, run, scores)
    post = json.loads(out.read_text())

    # Only the four target fields changed; everything else identical.
    for key in (
        "clip_id",
        "model",
        "lineage",
        "deployment",
        "prompts",
        "human_spot_checked",
        "human_spot_check_scores",
        "spot_check_disagreements",
    ):
        assert post[key] == pre[key], f"untouched field {key!r} should match pre-state"

    assert post["scores"] == {dim: (8 if dim != "ocr_fidelity" else None) for dim in _JUDGE_DIMS}
    assert post["composite"] == 7.42
    assert post["judge"] == "gpt-5"
    assert post["judge_justifications"] == {dim: f"j-{dim}" for dim in _JUDGE_DIMS}


def test_update_with_judge_error_leaves_scores_null(tmp_path: Path) -> None:
    out = _seed_json(tmp_path)
    run = JudgeRun(
        judge_run_id="rid-err",
        clip_id="smp-001",
        model="stub-adapter",
        judge_model="claude-opus-4-7",
        composite=None,
        error="judge response parse failure: invalid json",
        created_at="2026-04-30T18:00:00Z",
    )
    update_with_judge(out, run, [])
    post = json.loads(out.read_text())
    assert post["scores"] is None
    assert post["composite"] is None
    assert post["judge"] is None
    assert post["judge_justifications"] is None
    # prompts must still be intact.
    assert set(post["prompts"].keys()) == {"generic", "targeted", "hallucination_trap"}
