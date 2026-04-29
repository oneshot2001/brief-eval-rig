"""Output writer tests verify spec §9 schema compliance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brief_eval_rig.adapters.base import AnalysisResult, Clip, Deployment, Lineage, VLMAdapter
from brief_eval_rig.runner.output_writer import write_result


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
