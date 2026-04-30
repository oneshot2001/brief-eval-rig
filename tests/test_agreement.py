"""Tests for scoring.agreement (Cohen's kappa per dimension)."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from brief_eval_rig.scoring.agreement import (
    DIMENSIONS,
    AgreementRow,
    compute_all,
    compute_dimension_agreement,
    format_markdown_table,
)

SPOT_CHECK_SCHEMA = """
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


def _seed_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SPOT_CHECK_SCHEMA)
    rows = [
        # description_accuracy: 4 paired (8/7, 6/6, 9/9, 4/5), 1 with NULL human
        ("sc1", "r1", "description_accuracy", 8, 7, 1, None, "2026-04-30T20:00:00Z"),
        ("sc2", "r1", "description_accuracy", 6, 6, 0, None, "2026-04-30T20:00:01Z"),
        ("sc3", "r1", "description_accuracy", 9, 9, 0, None, "2026-04-30T20:00:02Z"),
        ("sc4", "r1", "description_accuracy", 4, 5, 1, None, "2026-04-30T20:00:03Z"),
        ("sc5", "r1", "description_accuracy", 7, None, None, "skip", "2026-04-30T20:00:04Z"),
        # ocr_fidelity: 2 paired, 1 with NULL judge
        ("sc6", "r1", "ocr_fidelity", 5, 6, 1, None, "2026-04-30T20:00:05Z"),
        ("sc7", "r1", "ocr_fidelity", 8, 8, 0, None, "2026-04-30T20:00:06Z"),
        ("sc8", "r1", "ocr_fidelity", None, 7, None, "N/A", "2026-04-30T20:00:07Z"),
    ]
    conn.executemany(
        """
        INSERT INTO spot_checks (
            spot_check_id, judge_run_id, dimension, judge_score, human_score,
            disagreement, notes, checked_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def test_perfect_agreement_kappa_one() -> None:
    row = compute_dimension_agreement([5, 7, 9, 3], [5, 7, 9, 3])
    assert row.n == 4
    assert row.kappa is not None
    assert row.kappa == pytest.approx(1.0)
    assert row.mean_disagreement == pytest.approx(0.0)


def test_partial_agreement_kappa_between_zero_and_one() -> None:
    row = compute_dimension_agreement([5, 7, 9, 3], [6, 6, 9, 4])
    assert row.n == 4
    assert row.kappa is not None
    assert 0.0 < row.kappa < 1.0
    assert row.mean_disagreement == pytest.approx(0.75)


def test_single_pair_kappa_none_disagreement_defined() -> None:
    row = compute_dimension_agreement([5], [7])
    assert row.n == 1
    assert row.kappa is None
    assert row.mean_disagreement == pytest.approx(2.0)


def test_empty_input_both_none() -> None:
    row = compute_dimension_agreement([], [])
    assert row.n == 0
    assert row.kappa is None
    assert row.mean_disagreement is None


def test_zero_variance_kappa_none_no_raise() -> None:
    row = compute_dimension_agreement([7, 7, 7], [7, 7, 7])
    assert row.n == 3
    assert row.kappa is None
    assert row.mean_disagreement == pytest.approx(0.0)


def test_compute_all_excludes_null_pairs() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        _seed_db(conn)
        rows = compute_all(conn)
    finally:
        conn.close()

    assert [r.dimension for r in rows] == list(DIMENSIONS)
    by_dim = {r.dimension: r for r in rows}

    assert by_dim["description_accuracy"].n == 4
    assert by_dim["description_accuracy"].kappa is not None
    assert by_dim["description_accuracy"].mean_disagreement == pytest.approx(0.5)

    assert by_dim["ocr_fidelity"].n == 2
    assert by_dim["ocr_fidelity"].mean_disagreement == pytest.approx(0.5)

    for dim in (
        "spatial_temporal",
        "targeted_query",
        "hallucination_resistance",
        "court_grade_language",
        "speed_cost",
    ):
        assert by_dim[dim].n == 0
        assert by_dim[dim].kappa is None
        assert by_dim[dim].mean_disagreement is None


def test_format_markdown_table_renders_em_dash_for_none() -> None:
    rows = [
        AgreementRow(dimension="description_accuracy", n=4, kappa=0.71, mean_disagreement=1.05),
        AgreementRow(dimension="ocr_fidelity", n=18, kappa=None, mean_disagreement=1.22),
        AgreementRow(dimension="speed_cost", n=0, kappa=None, mean_disagreement=None),
    ]
    table = format_markdown_table(rows)
    lines = table.splitlines()
    assert lines[0].startswith("|")
    assert "κ (judge vs human)" in lines[0]
    assert "mean disagreement" in lines[0]
    # row content
    assert "0.71" in lines[2]
    assert "1.05" in lines[2]
    assert "—" in lines[3]
    assert "1.22" in lines[3]
    # all-none row uses em-dash for both
    assert lines[4].count("—") == 2


def test_cli_smoke(tmp_path: Path) -> None:
    db_path = tmp_path / "scores.db"
    report_path = tmp_path / "reports" / "inter_rater_agreement.md"
    conn = sqlite3.connect(str(db_path))
    try:
        _seed_db(conn)
    finally:
        conn.close()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "brief_eval_rig.scoring.agreement",
            "--db",
            str(db_path),
            "--report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert report_path.exists()
    body = report_path.read_text(encoding="utf-8")
    assert "description_accuracy" in body
    assert "κ (judge vs human)" in body


def test_cli_missing_db_exits_one(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.db"
    report_path = tmp_path / "reports" / "out.md"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "brief_eval_rig.scoring.agreement",
            "--db",
            str(missing),
            "--report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "db file not found" in result.stderr


def test_cli_empty_db_prints_message(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.db"
    report_path = tmp_path / "reports" / "out.md"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SPOT_CHECK_SCHEMA)
        conn.commit()
    finally:
        conn.close()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "brief_eval_rig.scoring.agreement",
            "--db",
            str(db_path),
            "--report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "no spot-checks recorded yet" in result.stdout
    assert not report_path.exists()
