"""Tests for the interactive spot-check CLI."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from brief_eval_rig.scoring.rubric import DIMENSIONS
from brief_eval_rig.scoring.spot_check_cli import main, stratified_sample
from brief_eval_rig.scoring.storage import (
    Dimension,
    DimensionScore,
    JudgeRun,
    SpotCheck,
    connect,
    init_schema,
    record_spot_check,
    upsert_judge_run,
)

CLIPS = [
    ("ret-001", "retail"),
    ("ret-002", "retail"),
    ("com-001", "commercial"),
]
MODELS = ("alpha", "beta")


def _clip_payload(clip_id: str, vertical: str) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "vertical": vertical,
        "duration_seconds": 10.0,
        "resolution": "1920x1080",
        "framerate": 30,
        "scenario_tag": "test",
        "ground_truth_summary": f"summary for {clip_id}",
        "ground_truth_objects": ["person"],
        "ground_truth_ocr": [],
        "expected_temporal_events": [{"time": "0:01", "event": "enters"}],
        "targeted_query": "describe the subject",
        "hallucination_trap": "what color is the car",
        "difficulty": "easy",
        "source": "synthetic",
        "license": "internal-test-only",
        "minor_present": False,
        "consent_documented": True,
    }


def _output_payload(clip_id: str, model: str) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "model": model,
        "lineage": "anthropic",
        "deployment": "cloud",
        "prompts": {
            "generic": {
                "response": f"{model}-generic-{clip_id}",
                "latency_ms": 1,
                "cost_usd": 0.0,
                "error": None,
            },
            "targeted": {
                "response": f"{model}-targeted-{clip_id}",
                "latency_ms": 1,
                "cost_usd": 0.0,
                "error": None,
            },
            "hallucination_trap": {
                "response": f"{model}-trap-{clip_id}",
                "latency_ms": 1,
                "cost_usd": 0.0,
                "error": None,
            },
        },
        "scores": {dim: 8 for dim in DIMENSIONS},
        "composite": 8.0,
        "judge": "claude-opus-4-7",
        "judge_justifications": {dim: f"j-{dim}" for dim in DIMENSIONS},
        "human_spot_checked": False,
        "human_spot_check_scores": None,
        "spot_check_disagreements": [],
    }


@pytest.fixture
def env(tmp_path: Path) -> Iterator[dict[str, Any]]:
    corpus_dir = tmp_path / "corpus"
    outputs_dir = tmp_path / "outputs"
    db_path = outputs_dir / "scores.db"

    for clip_id, vertical in CLIPS:
        d = corpus_dir / vertical
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{clip_id}.json").write_text(
            json.dumps(_clip_payload(clip_id, vertical), indent=2) + "\n", encoding="utf-8"
        )

    for clip_id, _ in CLIPS:
        for model in MODELS:
            d = outputs_dir / clip_id
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{model}.json").write_text(
                json.dumps(_output_payload(clip_id, model), indent=2) + "\n", encoding="utf-8"
            )

    conn = connect(db_path)
    init_schema(conn)
    run_ids: dict[tuple[str, str], str] = {}
    for clip_id, _ in CLIPS:
        for model in MODELS:
            rid = uuid.uuid4().hex
            run = JudgeRun(
                judge_run_id=rid,
                clip_id=clip_id,
                model=model,
                judge_model="claude-opus-4-7",
                composite=8.0,
                error=None,
                judge_input_tokens=100,
                judge_output_tokens=50,
                judge_cost_usd=0.001,
                raw_judge_response="{}",
                created_at=f"2026-04-30T20:00:{len(run_ids):02d}Z",
            )
            scores = [
                DimensionScore(
                    judge_run_id=rid,
                    dimension=cast(Dimension, dim),
                    score=None if dim == "ocr_fidelity" else 8,
                    justification=f"j-{dim}",
                )
                for dim in DIMENSIONS
            ]
            upsert_judge_run(conn, run, scores)
            run_ids[(clip_id, model)] = rid
    conn.close()

    yield {
        "tmp_path": tmp_path,
        "corpus_dir": corpus_dir,
        "outputs_dir": outputs_dir,
        "db_path": db_path,
        "run_ids": run_ids,
    }


def _argv(env: dict[str, Any], *extra: str) -> list[str]:
    return [
        "--db",
        str(env["db_path"]),
        "--corpus",
        str(env["corpus_dir"]),
        "--outputs",
        str(env["outputs_dir"]),
        *extra,
    ]


def _session_path(env: dict[str, Any]) -> Path:
    outputs_dir: Path = env["outputs_dir"]
    return outputs_dir / ".spot_check_session.json"


def _read_session(env: dict[str, Any]) -> dict[str, Any]:
    raw: dict[str, Any] = json.loads(_session_path(env).read_text(encoding="utf-8"))
    return raw


def _make_input(answers: list[str]) -> Any:
    iterator = iter(answers)

    def fake_input(_prompt: str = "") -> str:
        try:
            return next(iterator)
        except StopIteration as exc:
            raise EOFError("ran out of fake inputs") from exc

    return fake_input


def _spot_checks_in_db(db_path: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute("SELECT * FROM spot_checks ORDER BY checked_at"))
    finally:
        conn.close()


def test_sample_creates_stratified_session(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("builtins.input", _make_input(["q"]))
    rc = main(_argv(env, "--sample", "4"))
    assert rc == 0
    session = _read_session(env)
    assert session["sample_size"] == 4
    assert len(session["judge_run_ids"]) == 4
    by_model: dict[str, int] = {}
    for rid in session["judge_run_ids"]:
        for (_, model), other_rid in env["run_ids"].items():
            if other_rid == rid:
                by_model[model] = by_model.get(model, 0) + 1
    assert by_model.get("alpha") == 2
    assert by_model.get("beta") == 2
    verticals_alpha = set()
    verticals_beta = set()
    for rid in session["judge_run_ids"]:
        for (clip_id, model), other_rid in env["run_ids"].items():
            if other_rid == rid:
                vert = next(v for c, v in CLIPS if c == clip_id)
                if model == "alpha":
                    verticals_alpha.add(vert)
                else:
                    verticals_beta.add(vert)
    assert verticals_alpha == {"retail", "commercial"}
    assert verticals_beta == {"retail", "commercial"}


def test_sample_aborts_when_session_exists(
    env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    _session_path(env).parent.mkdir(parents=True, exist_ok=True)
    _session_path(env).write_text("{}\n", encoding="utf-8")
    rc = main(_argv(env, "--sample", "4"))
    assert rc == 1
    err = capsys.readouterr().err
    assert "Existing spot-check session" in err
    assert "--resume" in err


def test_full_walk_blank_accepts_all(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    answers: list[str] = [""] * (4 * len(DIMENSIONS))
    monkeypatch.setattr("builtins.input", _make_input(answers))
    rc = main(_argv(env, "--sample", "4"))
    assert rc == 0
    session = _read_session(env)
    assert len(session["completed"]) == 4
    rows = _spot_checks_in_db(env["db_path"])
    assert len(rows) == 4 * len(DIMENSIONS)

    completed_ids = set(session["completed"])
    seen_jsons = 0
    for clip_id, _ in CLIPS:
        for model in MODELS:
            rid = env["run_ids"][(clip_id, model)]
            if rid not in completed_ids:
                continue
            seen_jsons += 1
            payload = json.loads(
                (env["outputs_dir"] / clip_id / f"{model}.json").read_text(encoding="utf-8")
            )
            assert payload["human_spot_checked"] is True
            assert payload["human_spot_check_scores"] is not None
            assert set(payload["human_spot_check_scores"].keys()) == set(DIMENSIONS)
    assert seen_jsons == 4


def test_override_records_disagreement(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    per_pair = ["3"] + [""] * (len(DIMENSIONS) - 1)
    answers = per_pair * 4
    monkeypatch.setattr("builtins.input", _make_input(answers))
    rc = main(_argv(env, "--sample", "4"))
    assert rc == 0
    rows = _spot_checks_in_db(env["db_path"])
    desc_rows = [r for r in rows if r["dimension"] == "description_accuracy"]
    assert len(desc_rows) == 4
    for r in desc_rows:
        assert r["human_score"] == 3
        assert r["judge_score"] == 8
        assert r["disagreement"] == 5

    session = _read_session(env)
    completed_ids = set(session["completed"])
    for clip_id, _ in CLIPS:
        for model in MODELS:
            rid = env["run_ids"][(clip_id, model)]
            if rid not in completed_ids:
                continue
            payload = json.loads(
                (env["outputs_dir"] / clip_id / f"{model}.json").read_text(encoding="utf-8")
            )
            assert "description_accuracy" in payload["spot_check_disagreements"]


def test_quit_mid_pair_does_not_record(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = ["", "", "", "q"]
    monkeypatch.setattr("builtins.input", _make_input(answers))
    rc = main(_argv(env, "--sample", "4"))
    assert rc == 0
    session = _read_session(env)
    assert session["completed"] == []
    rows = _spot_checks_in_db(env["db_path"])
    assert rows == []
    for clip_id, _ in CLIPS:
        for model in MODELS:
            payload = json.loads(
                (env["outputs_dir"] / clip_id / f"{model}.json").read_text(encoding="utf-8")
            )
            assert payload["human_spot_checked"] is False


def test_na_input_records_none(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    ocr_idx = DIMENSIONS.index("ocr_fidelity")
    per_pair: list[str] = []
    for i in range(len(DIMENSIONS)):
        per_pair.append("n" if i == ocr_idx else "")
    answers = per_pair * 4
    monkeypatch.setattr("builtins.input", _make_input(answers))
    rc = main(_argv(env, "--sample", "4"))
    assert rc == 0
    rows = _spot_checks_in_db(env["db_path"])
    ocr_rows = [r for r in rows if r["dimension"] == "ocr_fidelity"]
    assert len(ocr_rows) == 4
    for r in ocr_rows:
        assert r["human_score"] is None
        assert r["judge_score"] is None
        assert r["disagreement"] is None


def test_resume_walks_remaining(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    answers_first = [""] * (2 * len(DIMENSIONS)) + ["q"]
    monkeypatch.setattr("builtins.input", _make_input(answers_first))
    rc = main(_argv(env, "--sample", "4"))
    assert rc == 0
    session = _read_session(env)
    assert len(session["completed"]) == 2

    answers_second = [""] * (2 * len(DIMENSIONS))
    monkeypatch.setattr("builtins.input", _make_input(answers_second))
    rc = main(_argv(env, "--resume"))
    assert rc == 0
    session = _read_session(env)
    assert len(session["completed"]) == 4
    rows = _spot_checks_in_db(env["db_path"])
    assert len(rows) == 4 * len(DIMENSIONS)


def test_status_prints_counts_without_input(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    answers_first = [""] * (2 * len(DIMENSIONS)) + ["q"]
    monkeypatch.setattr("builtins.input", _make_input(answers_first))
    rc = main(_argv(env, "--sample", "4"))
    assert rc == 0
    capsys.readouterr()

    def boom(_prompt: str = "") -> str:
        raise AssertionError("status mode must not call input()")

    monkeypatch.setattr("builtins.input", boom)
    rc = main(_argv(env, "--status"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "Sample size: 4" in out
    assert "Completed:   2" in out
    assert "Remaining:   2" in out
    assert "Disagreements so far:" in out


def test_sampler_excludes_already_spot_checked_runs(env: dict[str, Any]) -> None:
    conn = connect(env["db_path"])
    init_schema(conn)
    excluded_rid = env["run_ids"][("ret-001", "alpha")]
    sc = SpotCheck(
        spot_check_id=uuid.uuid4().hex,
        judge_run_id=excluded_rid,
        dimension="description_accuracy",
        judge_score=8,
        human_score=8,
        disagreement=0,
        notes=None,
        checked_at="2026-04-30T18:00:00Z",
    )
    record_spot_check(conn, sc)
    conn.close()

    from brief_eval_rig.scoring.spot_check_cli import (
        _completed_judge_run_ids,
        _gradeable_runs,
        _vertical_map,
    )

    conn = connect(env["db_path"])
    try:
        runs = _gradeable_runs(conn)
        already = _completed_judge_run_ids(conn)
    finally:
        conn.close()
    assert excluded_rid in already
    vmap = _vertical_map(env["corpus_dir"])
    picks = stratified_sample(runs, vmap, already, 4)
    assert excluded_rid not in picks
