"""Tests for the judge CLI: discovery, dry-run, force, db override, summary."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import brief_eval_rig.scoring.judge as judge_cli
from brief_eval_rig.scoring.storage import DimensionScore, JudgeRun

_ALL_DIMS = (
    "description_accuracy",
    "ocr_fidelity",
    "spatial_temporal",
    "targeted_query",
    "hallucination_resistance",
    "court_grade_language",
    "speed_cost",
)


def _write_clip_meta(corpus_dir: Path, clip_id: str = "smp-001") -> None:
    sample = {
        "clip_id": clip_id,
        "vertical": "retail",
        "duration_seconds": 10.0,
        "resolution": "1920x1080",
        "framerate": 30,
        "scenario_tag": "loitering",
        "ground_truth_summary": "Subject enters and exits.",
        "ground_truth_objects": ["person"],
        "ground_truth_ocr": [],
        "expected_temporal_events": [
            {"time": "0:04", "event": "subject enters frame"},
            {"time": "0:23", "event": "subject exits frame"},
        ],
        "targeted_query": "Describe the person.",
        "hallucination_trap": "What does the license plate say?",
        "difficulty": "medium",
        "source": "synthetic",
        "license": "internal-test-only",
        "minor_present": False,
        "consent_documented": True,
    }
    sub = corpus_dir / "sample"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / f"{clip_id}.json").write_text(json.dumps(sample), encoding="utf-8")


def _write_model_output(
    outputs_dir: Path,
    *,
    clip_id: str = "smp-001",
    model: str = "qwen3-6-flash",
    lineage: str = "qwen",
    already_scored: bool = False,
) -> Path:
    payload: dict[str, Any] = {
        "clip_id": clip_id,
        "model": model,
        "lineage": lineage,
        "deployment": "cloud",
        "prompts": {
            kind: {
                "response": f"resp-{kind}",
                "latency_ms": 1000,
                "cost_usd": 0.01,
                "error": None,
            }
            for kind in ("generic", "targeted", "hallucination_trap")
        },
        "scores": ({dim: 5 for dim in _ALL_DIMS} if already_scored else None),
        "composite": 5.0 if already_scored else None,
        "judge": "claude-opus-4-7" if already_scored else None,
        "judge_justifications": ({dim: "j" for dim in _ALL_DIMS} if already_scored else None),
        "human_spot_checked": False,
        "human_spot_check_scores": None,
        "spot_check_disagreements": [],
    }
    clip_dir = outputs_dir / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    out = clip_dir / f"{model}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def _success_judge_run(
    clip_id: str, model: str, judge_model: str
) -> tuple[JudgeRun, list[DimensionScore]]:
    rid = f"rid-{clip_id}-{model}"
    run = JudgeRun(
        judge_run_id=rid,
        clip_id=clip_id,
        model=model,
        judge_model=judge_model,
        composite=7.5,
        error=None,
        judge_input_tokens=1000,
        judge_output_tokens=200,
        judge_cost_usd=0.0123,
        raw_judge_response='{"ok": true}',
        created_at="2026-04-30T18:00:00Z",
    )
    scores = [
        DimensionScore(
            judge_run_id=rid,
            dimension=dim,  # type: ignore[arg-type]
            score=8 if dim != "ocr_fidelity" else None,
            justification=f"j-{dim}",
        )
        for dim in _ALL_DIMS
    ]
    return run, scores


class _FakeJudge:
    def __init__(self) -> None:
        self.calls = 0

    def judge_pair(
        self,
        *,
        clip_metadata: Any,
        model_output: dict[str, Any],
        lineup_min_latency_ms: int,
        lineup_min_cost_usd: float,
    ) -> tuple[JudgeRun, list[DimensionScore]]:
        self.calls += 1
        clip_id = clip_metadata.clip_id
        model = str(model_output.get("model", ""))
        lineage = str(model_output.get("lineage", ""))
        judge_model = "gpt-5" if lineage == "anthropic" else "claude-opus-4-7"
        return _success_judge_run(clip_id, model, judge_model)


@pytest.fixture
def fake_judge(monkeypatch: pytest.MonkeyPatch) -> _FakeJudge:
    fj = _FakeJudge()
    monkeypatch.setattr(judge_cli, "_build_judge", lambda args: fj)
    return fj


def test_single_file_invocation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_judge: _FakeJudge
) -> None:
    corpus = tmp_path / "corpus"
    outputs = tmp_path / "outputs"
    db = tmp_path / "scores.db"
    _write_clip_meta(corpus)
    target = _write_model_output(outputs)

    rc = judge_cli.main([str(target), "--db", str(db), "--corpus", str(corpus)])
    assert rc == 0
    assert fake_judge.calls == 1
    out = capsys.readouterr().out
    assert "smp-001/qwen3-6-flash: judge=claude-opus-4-7" in out
    assert "graded 1 pairs" in out

    on_disk = json.loads(target.read_text())
    assert on_disk["judge"] == "claude-opus-4-7"
    assert on_disk["composite"] == 7.5
    # SQLite row exists.
    conn = sqlite3.connect(str(db))
    n = conn.execute("SELECT COUNT(*) FROM judge_runs").fetchone()[0]
    conn.close()
    assert n == 1


def test_directory_invocation_walks_clip_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_judge: _FakeJudge
) -> None:
    corpus = tmp_path / "corpus"
    outputs = tmp_path / "outputs"
    db = tmp_path / "scores.db"
    _write_clip_meta(corpus)
    _write_model_output(outputs, model="qwen3-6-flash", lineage="qwen")
    _write_model_output(outputs, model="claude-sonnet-4-6", lineage="anthropic")

    rc = judge_cli.main([str(outputs / "smp-001"), "--db", str(db), "--corpus", str(corpus)])
    assert rc == 0
    assert fake_judge.calls == 2
    out = capsys.readouterr().out
    assert "graded 2 pairs" in out


def test_dry_run_makes_no_calls_or_db(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_judge: _FakeJudge
) -> None:
    corpus = tmp_path / "corpus"
    outputs = tmp_path / "outputs"
    db = tmp_path / "scores.db"
    _write_clip_meta(corpus)
    target = _write_model_output(outputs)

    rc = judge_cli.main([str(outputs), "--dry-run", "--db", str(db), "--corpus", str(corpus)])
    assert rc == 0
    assert fake_judge.calls == 0
    out = capsys.readouterr().out
    assert "would grade smp-001/qwen3-6-flash" in out
    assert "estimated cost" in out
    # No DB created.
    assert not db.exists()
    # Original JSON untouched.
    on_disk = json.loads(target.read_text())
    assert on_disk["scores"] is None


def test_skip_already_graded_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_judge: _FakeJudge
) -> None:
    corpus = tmp_path / "corpus"
    outputs = tmp_path / "outputs"
    db = tmp_path / "scores.db"
    _write_clip_meta(corpus)
    _write_model_output(outputs, already_scored=True)

    rc = judge_cli.main([str(outputs), "--db", str(db), "--corpus", str(corpus)])
    assert rc == 0
    assert fake_judge.calls == 0
    out = capsys.readouterr().out
    assert "skip smp-001/qwen3-6-flash" in out
    assert "graded 0 pairs, skipped 1" in out


def test_force_regrades_already_scored(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_judge: _FakeJudge
) -> None:
    corpus = tmp_path / "corpus"
    outputs = tmp_path / "outputs"
    db = tmp_path / "scores.db"
    _write_clip_meta(corpus)
    _write_model_output(outputs, already_scored=True)

    rc = judge_cli.main([str(outputs), "--force", "--db", str(db), "--corpus", str(corpus)])
    assert rc == 0
    assert fake_judge.calls == 1
    out = capsys.readouterr().out
    assert "graded 1 pairs" in out


def test_db_override_path_used(tmp_path: Path, fake_judge: _FakeJudge) -> None:
    corpus = tmp_path / "corpus"
    outputs = tmp_path / "outputs"
    db = tmp_path / "custom.db"
    _write_clip_meta(corpus)
    _write_model_output(outputs)

    rc = judge_cli.main([str(outputs), "--db", str(db), "--corpus", str(corpus)])
    assert rc == 0
    assert db.exists()


def test_summary_line_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_judge: _FakeJudge
) -> None:
    corpus = tmp_path / "corpus"
    outputs = tmp_path / "outputs"
    db = tmp_path / "scores.db"
    _write_clip_meta(corpus)
    _write_model_output(outputs)

    rc = judge_cli.main([str(outputs), "--db", str(db), "--corpus", str(corpus)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "graded 1 pairs, skipped 0, errors 0, total cost $" in out
