"""End-to-end CLI tests for ``python -m brief_eval_rig.reporting.leaderboard``."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

# Phase 3 schema (mirrored from tests/test_aggregator.py — kept local so this
# test does not depend on the storage helper module).
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

CLIPS: dict[str, dict[str, str]] = {
    "tst-001": {"vertical": "retail", "difficulty": "medium"},
    "tst-002": {"vertical": "commercial", "difficulty": "hard"},
}

MODELS: tuple[str, ...] = ("mod-a", "mod-b")


def _build_corpus(corpus_dir: Path) -> None:
    """Write tmp ``corpus/{vertical}/{clip_id}.json`` sidecars."""
    for clip_id, spec in CLIPS.items():
        vert_dir = corpus_dir / spec["vertical"]
        vert_dir.mkdir(parents=True, exist_ok=True)
        sidecar = {
            "clip_id": clip_id,
            "vertical": spec["vertical"],
            "duration_seconds": 10.0,
            "resolution": "1920x1080",
            "framerate": 30,
            "scenario_tag": "synthetic",
            "ground_truth_summary": "synthetic",
            "ground_truth_objects": ["person"],
            "ground_truth_ocr": [],
            "expected_temporal_events": [],
            "targeted_query": "describe what happens",
            "hallucination_trap": "trap",
            "difficulty": spec["difficulty"],
            "source": "synthetic",
            "license": "internal-test-only",
            "minor_present": False,
            "consent_documented": True,
        }
        (vert_dir / f"{clip_id}.json").write_text(json.dumps(sidecar), encoding="utf-8")


def _build_outputs(outputs_dir: Path) -> None:
    """Write tmp ``outputs/{clip_id}/{model}.json`` per-clip payloads."""
    for clip_id in CLIPS:
        for model in MODELS:
            clip_dir = outputs_dir / clip_id
            clip_dir.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {
                "clip_id": clip_id,
                "model": model,
                "lineage": f"vendor-{model[-1]}",
                "deployment": "cloud",
                "prompts": {
                    "generic": {"latency_ms": 1000, "cost_usd": 0.01, "error": None},
                    "targeted": {"latency_ms": 1000, "cost_usd": 0.01, "error": None},
                    "hallucination_trap": {
                        "latency_ms": 1000,
                        "cost_usd": 0.01,
                        "error": None,
                    },
                },
                "scores": {},
                "composite": None,
                "judge": "claude-opus-4-7",
                "judge_justifications": {},
                "human_spot_checked": False,
                "human_spot_check_scores": None,
                "spot_check_disagreements": [],
            }
            (clip_dir / f"{model}.json").write_text(json.dumps(payload), encoding="utf-8")


def _seed_db(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA_SQL)
        for clip_id in CLIPS:
            for model in MODELS:
                run_id = uuid.uuid4().hex
                conn.execute(
                    """
                    INSERT INTO judge_runs (
                        judge_run_id, clip_id, model, judge_model, composite, error,
                        judge_input_tokens, judge_output_tokens, judge_cost_usd,
                        raw_judge_response, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        clip_id,
                        model,
                        "claude-opus-4-7",
                        7.0,
                        None,
                        100,
                        50,
                        0.001,
                        "{}",
                        "2026-04-30T18:00:00Z",
                    ),
                )
                # Score every dimension at 5 except description_accuracy varies
                # by model so leaderboard sort is deterministic. To make
                # difficulty-adjusted composite genuinely depend on the
                # difficulty weights, we also vary scores by clip — tst-001
                # (medium) gets +2, tst-002 (hard) gets -2 — so a different
                # easy/medium/hard weight blend yields a different weighted
                # average.
                desc_score = 8 if model == "mod-a" else 4
                clip_bias = 2 if clip_id == "tst-001" else -2
                for dim in DB_DIMENSIONS:
                    base = desc_score if dim == "description_accuracy" else 5
                    score = max(0, min(10, base + clip_bias))
                    conn.execute(
                        """
                        INSERT INTO dimension_scores
                        (judge_run_id, dimension, score, justification)
                        VALUES (?, ?, ?, ?)
                        """,
                        (run_id, dim, score, "synthetic"),
                    )
        conn.commit()
    finally:
        conn.close()


def _seed_empty_db(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def env(tmp_path: Path) -> Iterator[dict[str, Path]]:
    db_path = tmp_path / "scores.db"
    corpus_dir = tmp_path / "corpus"
    outputs_dir = tmp_path / "outputs"
    output_dir = tmp_path / "out"
    corpus_dir.mkdir()
    outputs_dir.mkdir()
    output_dir.mkdir()

    _build_corpus(corpus_dir)
    _build_outputs(outputs_dir)
    _seed_db(db_path)

    yield {
        "db": db_path,
        "corpus": corpus_dir,
        "outputs": outputs_dir,
        "output_dir": output_dir,
        "tmp_path": tmp_path,
    }


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "brief_eval_rig.reporting.leaderboard", *args],
        capture_output=True,
        text=True,
        check=False,
    )


# ---- Tests ---------------------------------------------------------------


def test_default_invocation_writes_three_artifacts(env: dict[str, Path]) -> None:
    result = _run_cli(
        "--db",
        str(env["db"]),
        "--output-dir",
        str(env["output_dir"]),
        "--corpus",
        str(env["corpus"]),
        "--outputs",
        str(env["outputs"]),
    )
    assert result.returncode == 0, result.stderr
    leaderboard = env["output_dir"] / "leaderboard.md"
    metrics = env["output_dir"] / "metrics.csv"
    reports_dir = env["output_dir"] / "reports"
    assert leaderboard.exists()
    assert metrics.exists()
    assert reports_dir.is_dir()
    for model in MODELS:
        assert (reports_dir / f"{model}.md").exists()


def test_db_not_found_exits_one(tmp_path: Path) -> None:
    result = _run_cli(
        "--db",
        str(tmp_path / "nonexistent.db"),
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 1
    assert "db file not found" in result.stderr


def test_empty_judge_runs_exits_two(tmp_path: Path) -> None:
    db_path = tmp_path / "scores.db"
    _seed_empty_db(db_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = _run_cli(
        "--db",
        str(db_path),
        "--output-dir",
        str(out_dir),
    )
    assert result.returncode == 2
    assert "nothing to report" in result.stderr


def test_difficulty_weights_parsing_changes_diff_adj(env: dict[str, Path]) -> None:
    """``--difficulty-weights`` parses correctly and the diff-adj composite changes."""
    out_default = env["output_dir"] / "default"
    out_custom = env["output_dir"] / "custom"
    out_default.mkdir()
    out_custom.mkdir()
    common = [
        "--db",
        str(env["db"]),
        "--corpus",
        str(env["corpus"]),
        "--outputs",
        str(env["outputs"]),
    ]
    r_default = _run_cli(*common, "--output-dir", str(out_default))
    r_custom = _run_cli(
        *common,
        "--output-dir",
        str(out_custom),
        "--difficulty-weights",
        "easy=0.1,medium=1.0,hard=2.0",
    )
    assert r_default.returncode == 0, r_default.stderr
    assert r_custom.returncode == 0, r_custom.stderr

    def _diff_adj(metrics_path: Path) -> dict[str, float]:
        text = metrics_path.read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(text))
        return {row["model"]: float(row["composite_difficulty_adjusted"]) for row in reader}

    default_diff = _diff_adj(out_default / "metrics.csv")
    custom_diff = _diff_adj(out_custom / "metrics.csv")
    # The diff-adj composite must change between weight configurations because
    # CLIPS contains both medium and hard difficulty pairs.
    assert default_diff != custom_diff


def test_two_runs_byte_identical(env: dict[str, Path]) -> None:
    out_a = env["output_dir"] / "run-a"
    out_b = env["output_dir"] / "run-b"
    out_a.mkdir()
    out_b.mkdir()
    common = [
        "--db",
        str(env["db"]),
        "--corpus",
        str(env["corpus"]),
        "--outputs",
        str(env["outputs"]),
    ]
    r1 = _run_cli(*common, "--output-dir", str(out_a))
    r2 = _run_cli(*common, "--output-dir", str(out_b))
    assert r1.returncode == 0 and r2.returncode == 0
    assert (out_a / "leaderboard.md").read_bytes() == (out_b / "leaderboard.md").read_bytes()
    assert (out_a / "metrics.csv").read_bytes() == (out_b / "metrics.csv").read_bytes()


def test_kappa_avg_appears_in_header(env: dict[str, Path]) -> None:
    result = _run_cli(
        "--db",
        str(env["db"]),
        "--output-dir",
        str(env["output_dir"]),
        "--corpus",
        str(env["corpus"]),
        "--outputs",
        str(env["outputs"]),
        "--kappa-avg",
        "0.74",
    )
    assert result.returncode == 0, result.stderr
    md = (env["output_dir"] / "leaderboard.md").read_text(encoding="utf-8")
    assert "κ averaged across dimensions: 0.74" in md


def test_min_n_suppresses_cells_but_models_still_listed(env: dict[str, Path]) -> None:
    """``--min-n 5`` with 1-pair-per-vertical data: dim/vertical cells em-dash; models listed."""
    result = _run_cli(
        "--db",
        str(env["db"]),
        "--output-dir",
        str(env["output_dir"]),
        "--corpus",
        str(env["corpus"]),
        "--outputs",
        str(env["outputs"]),
        "--min-n",
        "5",
    )
    assert result.returncode == 0, result.stderr
    md = (env["output_dir"] / "leaderboard.md").read_text(encoding="utf-8")
    overall_section = md.split("## Overall")[1].split("## By Dimension")[0]
    assert "| mod-a |" in overall_section
    assert "| mod-b |" in overall_section
    # Dim section: every numeric cell suppressed.
    dim_section = md.split("## By Dimension")[1].split("## By Vertical")[0]
    for model in MODELS:
        rows = [line for line in dim_section.splitlines() if line.startswith(f"| {model}")]
        assert len(rows) == 1
        cells = [c.strip() for c in rows[0].strip("|").split("|")]
        assert cells[1:8] == ["—"] * 7
