"""Tests for reporting.aggregator."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from brief_eval_rig.corpus.schema import ClipMetadata
from brief_eval_rig.reporting.aggregator import (
    DEFAULT_DIFFICULTY_WEIGHTS,
    VERTICALS,
    PerClipRow,
    aggregate_all_models,
    aggregate_one_model,
    load_per_clip_json,
)
from brief_eval_rig.scoring.rubric import compute_composite, compute_speed_cost

# Phase 3 schema (mirrored verbatim from scoring.storage so this test does not
# depend on storage helpers).
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS judge_runs (
    judge_run_id TEXT PRIMARY KEY,
    clip_id TEXT NOT NULL,
    model TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    composite REAL,
    error TEXT,
    judge_input_tokens INTEGER,
    judge_output_tokens INTEGER,
    judge_cost_usd REAL,
    raw_judge_response TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (clip_id, model)
);
CREATE TABLE IF NOT EXISTS dimension_scores (
    judge_run_id TEXT NOT NULL,
    dimension TEXT NOT NULL,
    score INTEGER,
    justification TEXT,
    PRIMARY KEY (judge_run_id, dimension)
);
CREATE TABLE IF NOT EXISTS spot_checks (
    spot_check_id TEXT PRIMARY KEY,
    judge_run_id TEXT NOT NULL,
    dimension TEXT NOT NULL,
    judge_score INTEGER,
    human_score INTEGER,
    disagreement INTEGER,
    notes TEXT,
    checked_at TEXT NOT NULL
);
"""

DB_DIMENSIONS: tuple[str, ...] = (
    "description_accuracy",
    "ocr_fidelity",
    "spatial_temporal",
    "targeted_query",
    "hallucination_resistance",
    "court_grade_language",
)

# ---- Fixture data --------------------------------------------------------

# Three clips. Total per-prompt latencies and costs are scoped per (model, clip).
CLIP_SPECS: dict[str, dict[str, Any]] = {
    "tst-001": {"vertical": "retail", "difficulty": "medium"},
    "tst-002": {"vertical": "retail", "difficulty": "hard"},
    "tst-003": {"vertical": "commercial", "difficulty": "easy"},
}

# Per-prompt latency_ms / cost_usd. Per-clip totals = sum of the 3 prompts.
# mod-a: $0.10/clip, slow.   mod-b: $0.05/clip (paid min), fast.   mod-c: free, mid.
MODEL_PROMPT_DATA: dict[str, dict[str, dict[str, dict[str, float]]]] = {
    "mod-a": {
        "tst-001": {
            "generic": {"latency_ms": 4000.0, "cost_usd": 0.04},
            "targeted": {"latency_ms": 3000.0, "cost_usd": 0.03},
            "hallucination_trap": {"latency_ms": 3000.0, "cost_usd": 0.03},
        },
        "tst-002": {
            "generic": {"latency_ms": 5000.0, "cost_usd": 0.04},
            "targeted": {"latency_ms": 3000.0, "cost_usd": 0.03},
            "hallucination_trap": {"latency_ms": 2000.0, "cost_usd": 0.03},
        },
        "tst-003": {
            "generic": {"latency_ms": 2000.0, "cost_usd": 0.04},
            "targeted": {"latency_ms": 4000.0, "cost_usd": 0.03},
            "hallucination_trap": {"latency_ms": 4000.0, "cost_usd": 0.03},
        },
    },
    "mod-b": {
        "tst-001": {
            "generic": {"latency_ms": 1000.0, "cost_usd": 0.02},
            "targeted": {"latency_ms": 1000.0, "cost_usd": 0.02},
            "hallucination_trap": {"latency_ms": 1000.0, "cost_usd": 0.01},
        },
        "tst-002": {
            "generic": {"latency_ms": 2000.0, "cost_usd": 0.02},
            "targeted": {"latency_ms": 1000.0, "cost_usd": 0.02},
            "hallucination_trap": {"latency_ms": 1000.0, "cost_usd": 0.01},
        },
        "tst-003": {
            "generic": {"latency_ms": 1500.0, "cost_usd": 0.02},
            "targeted": {"latency_ms": 1500.0, "cost_usd": 0.02},
            "hallucination_trap": {"latency_ms": 1000.0, "cost_usd": 0.01},
        },
    },
    "mod-c": {
        # mod-c is also the latency leader (matches mod-b), so its speed_cost
        # subscore averages out high (free pass on cost + ~10 on latency).
        "tst-001": {
            "generic": {"latency_ms": 1000.0, "cost_usd": 0.0},
            "targeted": {"latency_ms": 1000.0, "cost_usd": 0.0},
            "hallucination_trap": {"latency_ms": 1000.0, "cost_usd": 0.0},
        },
        "tst-002": {
            "generic": {"latency_ms": 2000.0, "cost_usd": 0.0},
            "targeted": {"latency_ms": 1000.0, "cost_usd": 0.0},
            "hallucination_trap": {"latency_ms": 1000.0, "cost_usd": 0.0},
        },
        "tst-003": {
            "generic": {"latency_ms": 1500.0, "cost_usd": 0.0},
            "targeted": {"latency_ms": 1500.0, "cost_usd": 0.0},
            "hallucination_trap": {"latency_ms": 1000.0, "cost_usd": 0.0},
        },
    },
}

# Synthetic 0/5/10 dim score mix.
MODEL_DIM_SCORES: dict[str, dict[str, dict[str, int | None]]] = {
    "mod-a": {
        "tst-001": {
            "description_accuracy": 5,
            "ocr_fidelity": None,
            "spatial_temporal": 5,
            "targeted_query": 5,
            "hallucination_resistance": 5,
            "court_grade_language": 5,
        },
        "tst-002": {
            "description_accuracy": 0,
            "ocr_fidelity": None,
            "spatial_temporal": 5,
            "targeted_query": 5,
            "hallucination_resistance": 5,
            "court_grade_language": 5,
        },
        "tst-003": {
            "description_accuracy": 10,
            "ocr_fidelity": None,
            "spatial_temporal": 5,
            "targeted_query": 5,
            "hallucination_resistance": 5,
            "court_grade_language": 5,
        },
    },
    "mod-b": {
        "tst-001": {
            "description_accuracy": 10,
            "ocr_fidelity": 10,
            "spatial_temporal": 10,
            "targeted_query": 10,
            "hallucination_resistance": 10,
            "court_grade_language": 10,
        },
        "tst-002": {
            "description_accuracy": 5,
            "ocr_fidelity": 5,
            "spatial_temporal": 5,
            "targeted_query": 5,
            "hallucination_resistance": 5,
            "court_grade_language": 5,
        },
        "tst-003": {
            "description_accuracy": 10,
            "ocr_fidelity": 10,
            "spatial_temporal": 10,
            "targeted_query": 10,
            "hallucination_resistance": 10,
            "court_grade_language": 10,
        },
    },
    "mod-c": {
        "tst-001": {
            "description_accuracy": 5,
            "ocr_fidelity": 5,
            "spatial_temporal": 5,
            "targeted_query": 5,
            "hallucination_resistance": 5,
            "court_grade_language": 5,
        },
        "tst-002": {
            "description_accuracy": 0,
            "ocr_fidelity": 0,
            "spatial_temporal": 0,
            "targeted_query": 0,
            "hallucination_resistance": 0,
            "court_grade_language": 0,
        },
        "tst-003": {
            "description_accuracy": 5,
            "ocr_fidelity": 5,
            "spatial_temporal": 5,
            "targeted_query": 5,
            "hallucination_resistance": 5,
            "court_grade_language": 5,
        },
    },
}

LINEAGE_DEPLOYMENT = {
    "mod-a": ("vendor-a", "cloud"),
    "mod-b": ("vendor-b", "cloud"),
    "mod-c": ("vendor-c", "local"),
}


def _build_clip_metadata(clip_id: str, vertical: str, difficulty: str) -> ClipMetadata:
    return ClipMetadata(
        clip_id=clip_id,
        vertical=vertical,  # type: ignore[arg-type]
        duration_seconds=10.0,
        resolution="1920x1080",
        framerate=30,
        scenario_tag="synthetic-test",
        ground_truth_summary="synthetic",
        ground_truth_objects=["person"],
        ground_truth_ocr=[],
        expected_temporal_events=[],
        targeted_query="describe what happens",
        hallucination_trap="trap",
        difficulty=difficulty,  # type: ignore[arg-type]
        source="synthetic",
        license="internal-test-only",
        minor_present=False,
        consent_documented=True,
    )


def _write_per_clip_json(outputs_dir: Path, clip_id: str, model: str) -> None:
    lineage, deployment = LINEAGE_DEPLOYMENT[model]
    payload: dict[str, Any] = {
        "clip_id": clip_id,
        "model": model,
        "lineage": lineage,
        "deployment": deployment,
        "prompts": {
            key: {
                "response": "synthetic",
                "latency_ms": MODEL_PROMPT_DATA[model][clip_id][key]["latency_ms"],
                "cost_usd": MODEL_PROMPT_DATA[model][clip_id][key]["cost_usd"],
                "error": None,
            }
            for key in ("generic", "targeted", "hallucination_trap")
        },
        "scores": {},
        "composite": None,
        "judge": "claude-opus-4-7",
        "judge_justifications": {},
        "human_spot_checked": False,
        "human_spot_check_scores": None,
        "spot_check_disagreements": [],
    }
    clip_dir = outputs_dir / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / f"{model}.json").write_text(json.dumps(payload), encoding="utf-8")


def _seed_judge_runs_and_scores(
    conn: sqlite3.Connection,
    *,
    error_for: tuple[str, str] | None = None,
    spot_check_for: tuple[str, str] | None = None,
) -> dict[tuple[str, str], str]:
    """Seed judge_runs + dimension_scores for every (model, clip) pair.

    Returns a map (model, clip_id) -> judge_run_id so callers can attach
    spot_checks deterministically.
    """
    run_ids: dict[tuple[str, str], str] = {}
    for model in MODEL_PROMPT_DATA:
        for clip_id in CLIP_SPECS:
            judge_run_id = uuid.uuid4().hex
            run_ids[(model, clip_id)] = judge_run_id
            error = None
            composite_value: float | None = 7.0
            if error_for == (model, clip_id):
                error = "synthetic-failure"
                composite_value = None
            conn.execute(
                """
                INSERT INTO judge_runs (
                    judge_run_id, clip_id, model, judge_model, composite, error,
                    judge_input_tokens, judge_output_tokens, judge_cost_usd,
                    raw_judge_response, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    judge_run_id,
                    clip_id,
                    model,
                    "claude-opus-4-7",
                    composite_value,
                    error,
                    100,
                    50,
                    0.001,
                    "{}",
                    "2026-04-30T18:00:00Z",
                ),
            )
            if error is not None:
                continue
            for dim, score in MODEL_DIM_SCORES[model][clip_id].items():
                conn.execute(
                    """
                    INSERT INTO dimension_scores (judge_run_id, dimension, score, justification)
                    VALUES (?, ?, ?, ?)
                    """,
                    (judge_run_id, dim, score, "synthetic"),
                )
            # Phase 3 also wrote a stale speed_cost row. Insert one per pair so
            # we can verify the aggregator ignores it.
            conn.execute(
                """
                INSERT INTO dimension_scores (judge_run_id, dimension, score, justification)
                VALUES (?, ?, ?, ?)
                """,
                (judge_run_id, "speed_cost", 0, "STALE-must-be-ignored"),
            )

    if spot_check_for is not None:
        target_run_id = run_ids[spot_check_for]
        conn.execute(
            """
            INSERT INTO spot_checks (
                spot_check_id, judge_run_id, dimension, judge_score,
                human_score, disagreement, notes, checked_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                target_run_id,
                "description_accuracy",
                5,
                5,
                0,
                None,
                "2026-04-30T20:00:00Z",
            ),
        )

    conn.commit()
    return run_ids


@pytest.fixture
def env(tmp_path: Path) -> Iterator[dict[str, Any]]:
    """Set up tmp DB + outputs/ + clip_lookup for an isolated aggregator run."""
    db_path = tmp_path / "scores.db"
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    conn.commit()

    for model in MODEL_PROMPT_DATA:
        for clip_id in CLIP_SPECS:
            _write_per_clip_json(outputs_dir, clip_id, model)

    clip_lookup = {
        clip_id: _build_clip_metadata(clip_id, spec["vertical"], spec["difficulty"])
        for clip_id, spec in CLIP_SPECS.items()
    }

    try:
        yield {
            "conn": conn,
            "outputs_dir": outputs_dir,
            "clip_lookup": clip_lookup,
        }
    finally:
        conn.close()


# ---- Helper computations -------------------------------------------------


def _expected_clip_totals(model: str, clip_id: str) -> tuple[int, float]:
    prompts = MODEL_PROMPT_DATA[model][clip_id]
    lat = sum(int(prompts[k]["latency_ms"]) for k in prompts)
    cost = sum(float(prompts[k]["cost_usd"]) for k in prompts)
    return lat, cost


def _expected_lineup_min(clip_id: str) -> tuple[int, float]:
    totals = [_expected_clip_totals(m, clip_id) for m in MODEL_PROMPT_DATA]
    min_lat = min(lat for lat, _ in totals)
    paid_costs = [c for _, c in totals if c > 0.0]
    min_paid_cost = min(paid_costs) if paid_costs else 0.0
    return min_lat, min_paid_cost


def _expected_per_clip_composite(model: str, clip_id: str) -> float:
    db_scores = dict(MODEL_DIM_SCORES[model][clip_id])
    clip_lat, clip_cost = _expected_clip_totals(model, clip_id)
    min_lat, min_paid_cost = _expected_lineup_min(clip_id)
    lat_sub, cost_sub = compute_speed_cost(
        latency_ms=clip_lat,
        cost_usd=clip_cost,
        lineup_min_latency_ms=min_lat,
        lineup_min_paid_cost_usd=min_paid_cost,
    )
    speed_cost_score = (lat_sub + cost_sub) / 2.0
    scores: dict[str, int | None] = dict(db_scores)
    scores["speed_cost"] = int(round(speed_cost_score))
    return compute_composite(scores)


# ---- Tests ---------------------------------------------------------------


def test_load_per_clip_json_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_per_clip_json(tmp_path, "missing-001", "mod-x") is None


def test_aggregate_one_model_n_clips_and_overall(env: dict[str, Any]) -> None:
    _seed_judge_runs_and_scores(env["conn"])
    agg = aggregate_one_model(
        env["conn"],
        "mod-a",
        clip_lookup=env["clip_lookup"],
        outputs_dir=env["outputs_dir"],
    )
    assert agg.n_clips == 3
    expected = [
        _expected_per_clip_composite("mod-a", "tst-001"),
        _expected_per_clip_composite("mod-a", "tst-002"),
        _expected_per_clip_composite("mod-a", "tst-003"),
    ]
    assert agg.composite_overall == pytest.approx(sum(expected) / 3.0)
    assert agg.lineage == "vendor-a"
    assert agg.deployment == "cloud"


def test_difficulty_adjusted_weighting(env: dict[str, Any]) -> None:
    _seed_judge_runs_and_scores(env["conn"])
    agg = aggregate_one_model(
        env["conn"],
        "mod-a",
        clip_lookup=env["clip_lookup"],
        outputs_dir=env["outputs_dir"],
    )
    weights = DEFAULT_DIFFICULTY_WEIGHTS  # easy=0.5, medium=1.0, hard=1.5
    c1 = _expected_per_clip_composite("mod-a", "tst-001")  # medium
    c2 = _expected_per_clip_composite("mod-a", "tst-002")  # hard
    c3 = _expected_per_clip_composite("mod-a", "tst-003")  # easy
    expected = (c1 * weights["medium"] + c2 * weights["hard"] + c3 * weights["easy"]) / (
        weights["medium"] + weights["hard"] + weights["easy"]
    )
    assert agg.composite_difficulty_adjusted == pytest.approx(expected)


def test_vertical_mean_retail(env: dict[str, Any]) -> None:
    _seed_judge_runs_and_scores(env["conn"])
    agg = aggregate_one_model(
        env["conn"],
        "mod-a",
        clip_lookup=env["clip_lookup"],
        outputs_dir=env["outputs_dir"],
    )
    c1 = _expected_per_clip_composite("mod-a", "tst-001")
    c2 = _expected_per_clip_composite("mod-a", "tst-002")
    assert agg.vertical_means["retail"] == pytest.approx((c1 + c2) / 2.0)


def test_vertical_mean_empty_is_none(env: dict[str, Any]) -> None:
    _seed_judge_runs_and_scores(env["conn"])
    agg = aggregate_one_model(
        env["conn"],
        "mod-a",
        clip_lookup=env["clip_lookup"],
        outputs_dir=env["outputs_dir"],
    )
    assert agg.vertical_means["schools"] is None
    assert set(agg.vertical_means.keys()) == set(VERTICALS)


def test_free_tier_speed_cost_normalization_excludes_free(env: dict[str, Any]) -> None:
    """Headline test: cost_subscore for paid models normalizes against the
    paid-only lineup min, not against the free-tier $0.

    With mod-a $0.10/clip, mod-b $0.05/clip (paid min), mod-c $0/clip:
      - mod-a per-pair cost_subscore = 10 * 0.05 / 0.10 = 5.0
      - mod-b per-pair cost_subscore = 10.0 (it IS the paid min)
      - mod-c per-pair cost_subscore = 10.0 (free-tier free pass)
    """
    _seed_judge_runs_and_scores(env["conn"])

    # Verify the rubric helper directly with the lineup minima we expect.
    for clip_id in CLIP_SPECS:
        min_lat, min_paid_cost = _expected_lineup_min(clip_id)
        assert min_paid_cost == pytest.approx(0.05)

        _, cost_sub_a = compute_speed_cost(
            latency_ms=sum(
                int(MODEL_PROMPT_DATA["mod-a"][clip_id][k]["latency_ms"])
                for k in MODEL_PROMPT_DATA["mod-a"][clip_id]
            ),
            cost_usd=0.10,
            lineup_min_latency_ms=min_lat,
            lineup_min_paid_cost_usd=min_paid_cost,
        )
        assert cost_sub_a == pytest.approx(5.0)

        _, cost_sub_b = compute_speed_cost(
            latency_ms=sum(
                int(MODEL_PROMPT_DATA["mod-b"][clip_id][k]["latency_ms"])
                for k in MODEL_PROMPT_DATA["mod-b"][clip_id]
            ),
            cost_usd=0.05,
            lineup_min_latency_ms=min_lat,
            lineup_min_paid_cost_usd=min_paid_cost,
        )
        assert cost_sub_b == pytest.approx(10.0)

        _, cost_sub_c = compute_speed_cost(
            latency_ms=sum(
                int(MODEL_PROMPT_DATA["mod-c"][clip_id][k]["latency_ms"])
                for k in MODEL_PROMPT_DATA["mod-c"][clip_id]
            ),
            cost_usd=0.0,
            lineup_min_latency_ms=min_lat,
            lineup_min_paid_cost_usd=min_paid_cost,
        )
        assert cost_sub_c == pytest.approx(10.0)

    # mod-c free-tier model: speed_cost dim mean should be high (>= 9).
    agg_c = aggregate_one_model(
        env["conn"],
        "mod-c",
        clip_lookup=env["clip_lookup"],
        outputs_dir=env["outputs_dir"],
    )
    assert agg_c.dim_means["speed_cost"] is not None
    assert agg_c.dim_means["speed_cost"] >= 9.0


def test_latency_p50(env: dict[str, Any]) -> None:
    _seed_judge_runs_and_scores(env["conn"])
    agg = aggregate_one_model(
        env["conn"],
        "mod-a",
        clip_lookup=env["clip_lookup"],
        outputs_dir=env["outputs_dir"],
    )
    expected_latencies: list[int] = []
    for clip_id in CLIP_SPECS:
        for k in MODEL_PROMPT_DATA["mod-a"][clip_id]:
            expected_latencies.append(int(MODEL_PROMPT_DATA["mod-a"][clip_id][k]["latency_ms"]))
    expected_latencies.sort()
    expected_p50 = float(expected_latencies[len(expected_latencies) // 2])
    assert agg.latency_ms_p50 == pytest.approx(expected_p50)


def test_cost_usd_total(env: dict[str, Any]) -> None:
    _seed_judge_runs_and_scores(env["conn"])
    agg = aggregate_one_model(
        env["conn"],
        "mod-a",
        clip_lookup=env["clip_lookup"],
        outputs_dir=env["outputs_dir"],
    )
    expected = 0.0
    for clip_id in CLIP_SPECS:
        for k in MODEL_PROMPT_DATA["mod-a"][clip_id]:
            expected += float(MODEL_PROMPT_DATA["mod-a"][clip_id][k]["cost_usd"])
    assert agg.cost_usd_total == pytest.approx(expected)


def test_judge_error_excluded_from_composite_overall(env: dict[str, Any]) -> None:
    _seed_judge_runs_and_scores(env["conn"], error_for=("mod-a", "tst-002"))
    agg = aggregate_one_model(
        env["conn"],
        "mod-a",
        clip_lookup=env["clip_lookup"],
        outputs_dir=env["outputs_dir"],
    )
    assert agg.n_with_errors == 1
    assert agg.n_clips == 2
    expected = [
        _expected_per_clip_composite("mod-a", "tst-001"),
        _expected_per_clip_composite("mod-a", "tst-003"),
    ]
    assert agg.composite_overall == pytest.approx(sum(expected) / 2.0)
    error_rows = [r for r in agg.per_clip if r.has_judge_error]
    assert len(error_rows) == 1
    assert error_rows[0].clip_id == "tst-002"


def test_n_spot_checked(env: dict[str, Any]) -> None:
    _seed_judge_runs_and_scores(env["conn"], spot_check_for=("mod-b", "tst-001"))
    agg_a = aggregate_one_model(
        env["conn"],
        "mod-a",
        clip_lookup=env["clip_lookup"],
        outputs_dir=env["outputs_dir"],
    )
    agg_b = aggregate_one_model(
        env["conn"],
        "mod-b",
        clip_lookup=env["clip_lookup"],
        outputs_dir=env["outputs_dir"],
    )
    assert agg_a.n_spot_checked == 0
    assert agg_b.n_spot_checked == 1


def test_per_clip_composite_known_input_known_output(env: dict[str, Any]) -> None:
    """Regression check: aggregator's per-pair composite must equal the
    spec composite formula applied to (DB dims 1-6 + fresh speed_cost int).

    The DB also stores a stale speed_cost=0 row that the aggregator must
    ignore — without that, this test fails.
    """
    _seed_judge_runs_and_scores(env["conn"])
    agg = aggregate_one_model(
        env["conn"],
        "mod-b",
        clip_lookup=env["clip_lookup"],
        outputs_dir=env["outputs_dir"],
    )
    by_clip: dict[str, PerClipRow] = {row.clip_id: row for row in agg.per_clip}
    for clip_id in CLIP_SPECS:
        expected = _expected_per_clip_composite("mod-b", clip_id)
        assert by_clip[clip_id].composite == pytest.approx(expected)


def test_aggregate_all_models_alphabetical(env: dict[str, Any]) -> None:
    _seed_judge_runs_and_scores(env["conn"])
    aggs = aggregate_all_models(
        env["conn"],
        clip_lookup=env["clip_lookup"],
        outputs_dir=env["outputs_dir"],
    )
    assert [a.model for a in aggs] == ["mod-a", "mod-b", "mod-c"]


def test_ocr_dim_mean_none_when_all_none(env: dict[str, Any]) -> None:
    """mod-a has every OCR row set to None; dim_means['ocr_fidelity'] is None."""
    _seed_judge_runs_and_scores(env["conn"])
    agg = aggregate_one_model(
        env["conn"],
        "mod-a",
        clip_lookup=env["clip_lookup"],
        outputs_dir=env["outputs_dir"],
    )
    assert agg.dim_means["ocr_fidelity"] is None
