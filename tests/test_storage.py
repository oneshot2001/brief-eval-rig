"""Round-trip tests for scoring.storage against in-memory SQLite."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from brief_eval_rig.scoring.storage import (
    Dimension,
    DimensionScore,
    JudgeRun,
    SpotCheck,
    connect,
    init_schema,
    iter_judge_runs,
    iter_spot_checks,
    record_spot_check,
    upsert_judge_run,
)

ALL_DIMENSIONS: tuple[Dimension, ...] = (
    "description_accuracy",
    "ocr_fidelity",
    "spatial_temporal",
    "targeted_query",
    "hallucination_resistance",
    "court_grade_language",
    "speed_cost",
)


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    c = connect(Path(":memory:"))
    init_schema(c)
    try:
        yield c
    finally:
        c.close()


def _make_run(clip_id: str = "ret-001", model: str = "claude-sonnet-4-6") -> JudgeRun:
    return JudgeRun(
        judge_run_id=uuid.uuid4().hex,
        clip_id=clip_id,
        model=model,
        judge_model="gpt-5",
        composite=7.42,
        error=None,
        judge_input_tokens=1234,
        judge_output_tokens=567,
        judge_cost_usd=0.0123,
        raw_judge_response='{"scores": {}}',
        created_at="2026-04-30T18:00:00Z",
    )


def _make_scores(run_id: str) -> list[DimensionScore]:
    return [
        DimensionScore(
            judge_run_id=run_id,
            dimension=dim,
            score=8 if dim != "ocr_fidelity" else None,
            justification=f"justification for {dim}",
        )
        for dim in ALL_DIMENSIONS
    ]


def test_init_schema_creates_all_tables(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    names = {row["name"] for row in rows}
    assert {"judge_runs", "dimension_scores", "spot_checks"}.issubset(names)


def test_upsert_judge_run_round_trip(conn: sqlite3.Connection) -> None:
    run = _make_run()
    scores = _make_scores(run.judge_run_id)
    upsert_judge_run(conn, run, scores)

    results = list(iter_judge_runs(conn))
    assert len(results) == 1
    got_run, got_scores = results[0]
    assert got_run == run
    assert sorted(got_scores, key=lambda s: s.dimension) == sorted(
        scores, key=lambda s: s.dimension
    )


def test_upsert_judge_run_idempotent_overwrite(conn: sqlite3.Connection) -> None:
    first = _make_run()
    first_scores = _make_scores(first.judge_run_id)
    upsert_judge_run(conn, first, first_scores)

    second = JudgeRun(
        judge_run_id=uuid.uuid4().hex,
        clip_id=first.clip_id,
        model=first.model,
        judge_model="claude-opus-4-7",
        composite=9.1,
        error=None,
        judge_input_tokens=999,
        judge_output_tokens=111,
        judge_cost_usd=0.05,
        raw_judge_response='{"scores": {"v": 2}}',
        created_at="2026-04-30T19:00:00Z",
    )
    second_scores = _make_scores(second.judge_run_id)
    upsert_judge_run(conn, second, second_scores)

    results = list(iter_judge_runs(conn))
    assert len(results) == 1
    got_run, got_scores = results[0]
    assert got_run == second
    assert all(s.judge_run_id == second.judge_run_id for s in got_scores)

    orphan_count = conn.execute(
        "SELECT COUNT(*) AS n FROM dimension_scores WHERE judge_run_id = ?",
        (first.judge_run_id,),
    ).fetchone()["n"]
    assert orphan_count == 0


def test_iter_judge_runs_filters(conn: sqlite3.Connection) -> None:
    run_a = _make_run(clip_id="ret-001", model="claude-sonnet-4-6")
    run_b = _make_run(clip_id="ret-002", model="gemini-2-5-flash")
    upsert_judge_run(conn, run_a, _make_scores(run_a.judge_run_id))
    upsert_judge_run(conn, run_b, _make_scores(run_b.judge_run_id))

    by_clip = list(iter_judge_runs(conn, clip_id="ret-002"))
    assert len(by_clip) == 1
    assert by_clip[0][0].clip_id == "ret-002"

    by_model = list(iter_judge_runs(conn, model="claude-sonnet-4-6"))
    assert len(by_model) == 1
    assert by_model[0][0].model == "claude-sonnet-4-6"


def test_record_spot_check_round_trip_and_filter(conn: sqlite3.Connection) -> None:
    run = _make_run()
    upsert_judge_run(conn, run, _make_scores(run.judge_run_id))

    sc1 = SpotCheck(
        spot_check_id=uuid.uuid4().hex,
        judge_run_id=run.judge_run_id,
        dimension="description_accuracy",
        judge_score=8,
        human_score=7,
        disagreement=1,
        notes="close call",
        checked_at="2026-04-30T20:00:00Z",
    )
    sc2 = SpotCheck(
        spot_check_id=uuid.uuid4().hex,
        judge_run_id=run.judge_run_id,
        dimension="ocr_fidelity",
        judge_score=None,
        human_score=None,
        disagreement=None,
        notes="N/A",
        checked_at="2026-04-30T20:01:00Z",
    )
    record_spot_check(conn, sc1)
    record_spot_check(conn, sc2)

    all_checks = list(iter_spot_checks(conn))
    assert len(all_checks) == 2
    assert {c.spot_check_id for c in all_checks} == {sc1.spot_check_id, sc2.spot_check_id}

    only_ocr = list(iter_spot_checks(conn, dimension="ocr_fidelity"))
    assert len(only_ocr) == 1
    assert only_ocr[0] == sc2


def test_foreign_key_cascade_on_delete(conn: sqlite3.Connection) -> None:
    run = _make_run()
    upsert_judge_run(conn, run, _make_scores(run.judge_run_id))
    sc = SpotCheck(
        spot_check_id=uuid.uuid4().hex,
        judge_run_id=run.judge_run_id,
        dimension="targeted_query",
        judge_score=6,
        human_score=6,
        disagreement=0,
        notes=None,
        checked_at="2026-04-30T20:02:00Z",
    )
    record_spot_check(conn, sc)

    with conn:
        conn.execute("DELETE FROM judge_runs WHERE judge_run_id = ?", (run.judge_run_id,))

    ds_left = conn.execute(
        "SELECT COUNT(*) AS n FROM dimension_scores WHERE judge_run_id = ?",
        (run.judge_run_id,),
    ).fetchone()["n"]
    sc_left = conn.execute(
        "SELECT COUNT(*) AS n FROM spot_checks WHERE judge_run_id = ?",
        (run.judge_run_id,),
    ).fetchone()["n"]
    assert ds_left == 0
    assert sc_left == 0
