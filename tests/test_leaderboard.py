"""Tests for reporting.leaderboard render functions (pure, no I/O)."""

from __future__ import annotations

import csv
import io

import pytest

from brief_eval_rig.reporting.aggregator import (
    VERTICALS,
    ModelAggregate,
    PerClipRow,
)
from brief_eval_rig.reporting.leaderboard import (
    CSV_COLUMNS,
    render_leaderboard_md,
    render_metrics_csv,
    render_per_model_report_md,
)
from brief_eval_rig.scoring.rubric import DIMENSIONS


def _agg(
    *,
    model: str,
    composite: float = 5.0,
    diff_adj: float | None = None,
    n_clips: int = 3,
    n_with_errors: int = 0,
    lineage: str = "vendor-x",
    deployment: str = "cloud",
    dim_means: dict[str, float | None] | None = None,
    vertical_means: dict[str, float | None] | None = None,
    latency_ms_p50: float = 1500.0,
    cost_usd_total: float = 0.0123,
    n_spot_checked: int = 0,
    per_clip: tuple[PerClipRow, ...] | None = None,
) -> ModelAggregate:
    """Build a synthetic ModelAggregate for renderer tests."""
    if dim_means is None:
        dim_means = {dim: 5.0 for dim in DIMENSIONS}
    else:
        # Backfill any missing dim with 5.0 so tests can supply only the
        # dimensions they care about.
        dim_means = {dim: dim_means.get(dim, 5.0) for dim in DIMENSIONS}
    if vertical_means is None:
        vertical_means = {v: None for v in VERTICALS}
        vertical_means["retail"] = composite
    else:
        vertical_means = {v: vertical_means.get(v) for v in VERTICALS}
    if per_clip is None:
        per_clip = (
            PerClipRow(
                clip_id="tst-001",
                vertical="retail",
                difficulty="medium",
                composite=composite,
                latency_ms_total=int(latency_ms_p50 * 3),
                cost_usd_total=cost_usd_total,
                has_judge_error=False,
            ),
        )
    return ModelAggregate(
        model=model,
        lineage=lineage,
        deployment=deployment,
        n_clips=n_clips,
        n_with_errors=n_with_errors,
        composite_overall=composite,
        composite_difficulty_adjusted=diff_adj if diff_adj is not None else composite,
        dim_means=dim_means,
        vertical_means=vertical_means,
        latency_ms_p50=latency_ms_p50,
        cost_usd_total=cost_usd_total,
        n_spot_checked=n_spot_checked,
        per_clip=per_clip,
    )


# ---- render_leaderboard_md ----------------------------------------------


def test_leaderboard_has_three_sections_and_methodology() -> None:
    aggs = [
        _agg(model="alpha", composite=7.0),
        _agg(model="bravo", composite=5.0),
        _agg(model="charlie", composite=3.0),
    ]
    md = render_leaderboard_md(aggs)
    assert "# Brief Eval Rig — Leaderboard" in md
    assert "## Overall" in md
    assert "## By Dimension" in md
    assert "## By Vertical" in md
    assert "## Methodology" in md
    assert "— end —" in md


def test_leaderboard_data_row_counts() -> None:
    aggs = [
        _agg(model="alpha", composite=7.0),
        _agg(model="bravo", composite=5.0),
        _agg(model="charlie", composite=3.0),
    ]
    md = render_leaderboard_md(aggs)
    # Each table has exactly 3 data rows. Count "| alpha", "| bravo", "| charlie".
    for name in ("alpha", "bravo", "charlie"):
        # Three tables (Overall + Dim + Vertical) each list every model once.
        assert md.count(f"| {name} ") == 3


def test_leaderboard_is_deterministic() -> None:
    aggs = [_agg(model="alpha", composite=7.0), _agg(model="bravo", composite=5.0)]
    a = render_leaderboard_md(aggs)
    b = render_leaderboard_md(aggs)
    assert a == b


def test_leaderboard_sort_order_by_composite() -> None:
    aggs = [
        _agg(model="a", composite=5.0),
        _agg(model="b", composite=8.0),
        _agg(model="c", composite=3.0),
    ]
    md = render_leaderboard_md(aggs)
    # In the Overall table, b appears before a, which appears before c.
    overall_section = md.split("## Overall")[1].split("## By Dimension")[0]
    pos_a = overall_section.find("| a |")
    pos_b = overall_section.find("| b |")
    pos_c = overall_section.find("| c |")
    assert 0 < pos_b < pos_a < pos_c


def test_leaderboard_none_dim_renders_em_dash() -> None:
    agg = _agg(model="alpha", composite=7.0, dim_means={"ocr_fidelity": None})
    md = render_leaderboard_md([agg])
    # Find the dim row for alpha. The OCR column is the 3rd cell after model.
    dim_section = md.split("## By Dimension")[1].split("## By Vertical")[0]
    assert "—" in dim_section
    # Specifically: the row contains an em-dash where OCR would go.
    rows = [line for line in dim_section.splitlines() if line.startswith("| alpha")]
    assert len(rows) == 1
    # OCR is the second dim column in DIMENSIONS order.
    cells = [c.strip() for c in rows[0].strip("|").split("|")]
    # cells[0] = "alpha", cells[1] = description_accuracy, cells[2] = ocr_fidelity.
    assert cells[2] == "—"


def test_leaderboard_error_footnote_and_star() -> None:
    aggs = [
        _agg(model="alpha", composite=7.0, n_with_errors=2, n_clips=4),
        _agg(model="bravo", composite=5.0, n_with_errors=0),
    ]
    md = render_leaderboard_md(aggs)
    # Star on alpha in Overall table
    overall_section = md.split("## Overall")[1].split("## By Dimension")[0]
    assert "| alpha* |" in overall_section
    assert "| bravo |" in overall_section
    assert "*upstream prompt errors on" in md


def test_leaderboard_tie_break_alphabetical() -> None:
    aggs = [
        _agg(model="zeta", composite=7.0),
        _agg(model="alpha", composite=7.0),
        _agg(model="mu", composite=7.0),
    ]
    md = render_leaderboard_md(aggs)
    overall_section = md.split("## Overall")[1].split("## By Dimension")[0]
    pos_alpha = overall_section.find("| alpha |")
    pos_mu = overall_section.find("| mu |")
    pos_zeta = overall_section.find("| zeta |")
    assert 0 < pos_alpha < pos_mu < pos_zeta


def test_leaderboard_kappa_default_em_dash() -> None:
    md = render_leaderboard_md([_agg(model="alpha")])
    assert "κ averaged across dimensions: —" in md


def test_leaderboard_kappa_value() -> None:
    md = render_leaderboard_md([_agg(model="alpha")], kappa_average=0.74)
    assert "κ averaged across dimensions: 0.74" in md


# ---- render_per_model_report_md -----------------------------------------


def test_per_model_strengths_top_three() -> None:
    dim_means: dict[str, float | None] = {
        "description_accuracy": 9.0,
        "ocr_fidelity": 7.5,
        "spatial_temporal": 5.0,
        "targeted_query": 8.5,
        "hallucination_resistance": 6.0,
        "court_grade_language": 4.0,
        "speed_cost": 3.0,
    }
    agg = _agg(model="alpha", dim_means=dim_means)
    md = render_per_model_report_md(agg)
    strengths_section = md.split("## Strengths")[1].split("## Weaknesses")[0]
    # Top 3: description_accuracy 9.0, targeted_query 8.5, ocr_fidelity 7.5
    assert "description_accuracy" in strengths_section
    assert "targeted_query" in strengths_section
    assert "ocr_fidelity" in strengths_section
    assert "speed_cost" not in strengths_section


def test_per_model_weaknesses_bottom_three() -> None:
    dim_means: dict[str, float | None] = {
        "description_accuracy": 9.0,
        "ocr_fidelity": 7.5,
        "spatial_temporal": 5.0,
        "targeted_query": 8.5,
        "hallucination_resistance": 6.0,
        "court_grade_language": 4.0,
        "speed_cost": 3.0,
    }
    agg = _agg(model="alpha", dim_means=dim_means)
    md = render_per_model_report_md(agg)
    weaknesses_section = md.split("## Weaknesses")[1].split("## Per-Vertical")[0]
    # Bottom 3: speed_cost 3.0, court_grade_language 4.0, spatial_temporal 5.0
    assert "speed_cost" in weaknesses_section
    assert "court_grade_language" in weaknesses_section
    assert "spatial_temporal" in weaknesses_section
    assert "description_accuracy" not in weaknesses_section


def test_per_model_strengths_excludes_none() -> None:
    dim_means: dict[str, float | None] = {
        "description_accuracy": 9.0,
        "ocr_fidelity": None,
        "spatial_temporal": 5.0,
        "targeted_query": 8.5,
        "hallucination_resistance": 6.0,
        "court_grade_language": 4.0,
        "speed_cost": 3.0,
    }
    agg = _agg(model="alpha", dim_means=dim_means)
    md = render_per_model_report_md(agg)
    strengths_section = md.split("## Strengths")[1].split("## Weaknesses")[0]
    weaknesses_section = md.split("## Weaknesses")[1].split("## Per-Vertical")[0]
    # ocr_fidelity is None → must not appear in either section.
    assert "ocr_fidelity" not in strengths_section
    assert "ocr_fidelity" not in weaknesses_section


def test_per_model_per_clip_sorted_and_error_marker() -> None:
    rows = (
        PerClipRow(
            clip_id="tst-003",
            vertical="retail",
            difficulty="easy",
            composite=6.0,
            latency_ms_total=2000,
            cost_usd_total=0.01,
            has_judge_error=False,
        ),
        PerClipRow(
            clip_id="tst-001",
            vertical="retail",
            difficulty="medium",
            composite=0.0,
            latency_ms_total=0,
            cost_usd_total=0.0,
            has_judge_error=True,
        ),
        PerClipRow(
            clip_id="tst-002",
            vertical="retail",
            difficulty="hard",
            composite=7.0,
            latency_ms_total=3000,
            cost_usd_total=0.02,
            has_judge_error=False,
        ),
    )
    agg = _agg(model="alpha", per_clip=rows)
    md = render_per_model_report_md(agg)
    detail_section = md.split("## Per-Clip Detail")[1]
    pos_001 = detail_section.find("| tst-001 |")
    pos_002 = detail_section.find("| tst-002 |")
    pos_003 = detail_section.find("| tst-003 |")
    assert 0 < pos_001 < pos_002 < pos_003
    assert "❌" in detail_section


# ---- render_metrics_csv -------------------------------------------------


def test_csv_header_matches_columns_constant() -> None:
    csv_text = render_metrics_csv([_agg(model="alpha")])
    reader = csv.reader(io.StringIO(csv_text))
    header = next(reader)
    assert tuple(header) == CSV_COLUMNS


def test_csv_data_row_count() -> None:
    aggs = [
        _agg(model="alpha", composite=7.0),
        _agg(model="bravo", composite=5.0),
        _agg(model="charlie", composite=3.0),
    ]
    csv_text = render_metrics_csv(aggs)
    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)
    # 1 header + N data rows
    assert len(rows) == 4


def test_csv_none_renders_empty_string() -> None:
    agg = _agg(
        model="alpha",
        dim_means={"ocr_fidelity": None},
        vertical_means={"retail": 7.0},
    )
    csv_text = render_metrics_csv([agg])
    reader = csv.DictReader(io.StringIO(csv_text))
    row = next(reader)
    assert row["dim_ocr_fidelity_mean"] == ""
    # Verticals not provided also render as empty.
    assert row["vertical_commercial_mean"] == ""


def test_csv_float_precision_four_decimals() -> None:
    agg = _agg(model="alpha", composite=7.123456789, cost_usd_total=0.012345)
    csv_text = render_metrics_csv([agg])
    reader = csv.DictReader(io.StringIO(csv_text))
    row = next(reader)
    assert row["composite_overall"] == "7.1235"
    assert row["cost_usd_total"] == "0.0123"


def test_csv_sort_order_matches_leaderboard() -> None:
    aggs = [
        _agg(model="alpha", composite=5.0),
        _agg(model="zeta", composite=8.0),
        _agg(model="mu", composite=8.0),  # Tied with zeta; alpha-first
        _agg(model="charlie", composite=3.0),
    ]
    csv_text = render_metrics_csv(aggs)
    reader = csv.DictReader(io.StringIO(csv_text))
    models = [row["model"] for row in reader]
    assert models == ["mu", "zeta", "alpha", "charlie"]


def test_render_metrics_csv_is_deterministic() -> None:
    aggs = [_agg(model="alpha", composite=7.0), _agg(model="bravo", composite=5.0)]
    a = render_metrics_csv(aggs)
    b = render_metrics_csv(aggs)
    assert a == b


# ---- min-n suppression ----------------------------------------------------


def test_min_n_suppresses_dim_and_vertical_cells() -> None:
    """With min_n=5 and n_clips=1, dim/vertical cells render as em-dash."""
    agg = _agg(model="alpha", n_clips=1, composite=7.0)
    md = render_leaderboard_md([agg], min_n=5)
    # Overall row still present with composite.
    overall_section = md.split("## Overall")[1].split("## By Dimension")[0]
    assert "| alpha |" in overall_section
    # Dim row exists but every cell is em-dash.
    dim_section = md.split("## By Dimension")[1].split("## By Vertical")[0]
    rows = [line for line in dim_section.splitlines() if line.startswith("| alpha")]
    assert len(rows) == 1
    cells = [c.strip() for c in rows[0].strip("|").split("|")]
    # cells[0] = model, cells[1..7] = 7 dimensions; all 7 should be em-dash.
    assert cells[1:8] == ["—"] * 7
    # Vertical row should also be all em-dashes.
    vert_section = md.split("## By Vertical")[1].split("## Methodology")[0]
    vert_rows = [line for line in vert_section.splitlines() if line.startswith("| alpha")]
    assert len(vert_rows) == 1
    vcells = [c.strip() for c in vert_rows[0].strip("|").split("|")]
    assert vcells[1 : 1 + len(VERTICALS)] == ["—"] * len(VERTICALS)


def test_min_n_default_one_does_not_suppress() -> None:
    agg = _agg(model="alpha", n_clips=1, composite=7.0)
    md = render_leaderboard_md([agg], min_n=1)
    dim_section = md.split("## By Dimension")[1].split("## By Vertical")[0]
    rows = [line for line in dim_section.splitlines() if line.startswith("| alpha")]
    cells = [c.strip() for c in rows[0].strip("|").split("|")]
    # At least one numeric (non-em-dash) cell should be present.
    assert any(c not in ("—", "alpha", "") for c in cells)


# Sanity: pytest import works.
def test_pytest_smoke() -> None:
    assert pytest is not None
