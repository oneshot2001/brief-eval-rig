"""Phase 4 leaderboard generator.

Renders ``ModelAggregate`` lists to ``leaderboard.md``, ``reports/{model}.md``,
and ``metrics.csv``.

The three render functions are pure (no I/O) so they're trivial to unit-test;
the CLI ``main()`` is the only function that touches the filesystem or the
SQLite DB. Output is deterministic given identical input — the only
non-data-driven piece is the single ``Generated: YYYY-MM-DD`` header line, which
uses ``date.today().isoformat()`` (deterministic per day, not per call).
"""

from __future__ import annotations

import argparse
import csv
import io
import sqlite3
import sys
from collections.abc import Mapping
from datetime import date
from pathlib import Path

from brief_eval_rig.reporting.aggregator import (
    DEFAULT_DIFFICULTY_WEIGHTS,
    VERTICALS,
    ModelAggregate,
    aggregate_all_models,
    load_clip_lookup,
)
from brief_eval_rig.scoring.rubric import DIMENSIONS

DIM_DISPLAY: dict[str, str] = {
    "description_accuracy": "Description",
    "ocr_fidelity": "OCR",
    "spatial_temporal": "Spatial",
    "targeted_query": "Targeted",
    "hallucination_resistance": "Hallucination",
    "court_grade_language": "Court-Grade",
    "speed_cost": "Speed/Cost",
}

# Fixed CSV column order — downstream charts hard-code these names. Do not
# reorder without bumping the rig version and updating all consumers.
CSV_COLUMNS: tuple[str, ...] = (
    "model",
    "lineage",
    "deployment",
    "n_clips",
    "n_with_errors",
    "n_spot_checked",
    "composite_overall",
    "composite_difficulty_adjusted",
    "dim_description_accuracy_mean",
    "dim_ocr_fidelity_mean",
    "dim_spatial_temporal_mean",
    "dim_targeted_query_mean",
    "dim_hallucination_resistance_mean",
    "dim_court_grade_language_mean",
    "dim_speed_cost_mean",
    "vertical_retail_mean",
    "vertical_commercial_mean",
    "vertical_construction_mean",
    "vertical_schools_mean",
    "vertical_industrial_mean",
    "vertical_residential_mean",
    "vertical_edge_cases_mean",
    "vertical_long_form_mean",
    "latency_ms_p50",
    "cost_usd_total",
)

EM_DASH = "—"


def _sorted_aggregates(aggregates: list[ModelAggregate]) -> list[ModelAggregate]:
    """Composite-desc, model-name-asc tiebreak. Used everywhere we render a table."""
    return sorted(aggregates, key=lambda a: (-a.composite_overall, a.model))


def _fmt_float(value: float | None, *, places: int = 2) -> str:
    if value is None:
        return EM_DASH
    return f"{value:.{places}f}"


def _fmt_latency(latency_ms: float | None) -> str:
    """Render latency as ``Xs`` if ≥1000ms, else ``Xms``. Em-dash when None."""
    if latency_ms is None:
        return EM_DASH
    if latency_ms >= 1000.0:
        return f"{latency_ms / 1000.0:.2f}s"
    return f"{latency_ms:.0f}ms"


def _fmt_cost(cost_usd: float | None, *, places: int = 4) -> str:
    if cost_usd is None:
        return EM_DASH
    return f"${cost_usd:.{places}f}"


def _suppressed_dim_means(agg: ModelAggregate, *, min_n: int) -> dict[str, float | None]:
    """Apply ``--min-n`` suppression to per-dimension means.

    A dimension cell is suppressed when fewer than ``min_n`` graded pairs
    contributed a non-None score to its mean. The aggregator does not store
    per-dim n explicitly; we approximate using ``n_clips`` because for this
    rig every graded pair contributes to every dim except OCR (which uses
    None to signal "no OCR ground truth"). For ``min_n == 1`` (default), this
    is a no-op.
    """
    if min_n <= 1:
        return dict(agg.dim_means)
    out: dict[str, float | None] = {}
    for dim, value in agg.dim_means.items():
        if value is None or agg.n_clips < min_n:
            out[dim] = None
        else:
            out[dim] = value
    return out


def _suppressed_vertical_means(agg: ModelAggregate, *, min_n: int) -> dict[str, float | None]:
    """Apply ``--min-n`` suppression to per-vertical means.

    A vertical cell is suppressed when fewer than ``min_n`` graded pairs in
    that vertical contributed to its mean. The aggregator stores per-vertical
    n implicitly via ``per_clip``; we count rows per vertical here.
    """
    if min_n <= 1:
        return dict(agg.vertical_means)
    counts: dict[str, int] = dict.fromkeys(VERTICALS, 0)
    for row in agg.per_clip:
        if row.has_judge_error:
            continue
        if row.vertical in counts:
            counts[row.vertical] += 1
    out: dict[str, float | None] = {}
    for vert, value in agg.vertical_means.items():
        if value is None or counts.get(vert, 0) < min_n:
            out[vert] = None
        else:
            out[vert] = value
    return out


# ---- Markdown leaderboard ------------------------------------------------


def render_leaderboard_md(
    aggregates: list[ModelAggregate],
    *,
    kappa_average: float | None = None,
    min_n: int = 1,
) -> str:
    """Render the public leaderboard as Markdown.

    Three sections (Overall / By Dimension / By Vertical) plus a Methodology
    footer. Sort order is composite descending with ties broken alphabetically
    by model name.
    """
    sorted_aggs = _sorted_aggregates(aggregates)
    n_models = len(sorted_aggs)
    n_clips_max = max((a.n_clips for a in sorted_aggs), default=0)
    kappa_str = _fmt_float(kappa_average, places=2)
    today = date.today().isoformat()

    has_error_footnote = any(a.n_with_errors > 0 for a in sorted_aggs)

    lines: list[str] = []
    lines.append("# Brief Eval Rig — Leaderboard")
    lines.append("")
    lines.append(
        f"Generated: {today} · n_models: {n_models} · n_clips_max: {n_clips_max} · "
        f"κ averaged across dimensions: {kappa_str}"
    )
    lines.append("")

    # Overall
    lines.append("## Overall")
    lines.append("")
    lines.append(
        "| Rank | Model | Lineage | Deployment | n | Composite | Diff-Adj "
        "| Latency p50 | Cost total |"
    )
    lines.append(
        "|-----:|-------|---------|------------|--:|----------:|---------:"
        "|------------:|-----------:|"
    )
    for rank, agg in enumerate(sorted_aggs, start=1):
        model_name = agg.model + ("*" if agg.n_with_errors > 0 else "")
        composite = _fmt_float(agg.composite_overall)
        diff_adj = _fmt_float(agg.composite_difficulty_adjusted)
        latency = _fmt_latency(agg.latency_ms_p50)
        cost = _fmt_cost(agg.cost_usd_total)
        lineage = agg.lineage or EM_DASH
        deployment = agg.deployment or EM_DASH
        lines.append(
            f"| {rank} | {model_name} | {lineage} | {deployment} | "
            f"{agg.n_clips} | {composite} | {diff_adj} | {latency} | {cost} |"
        )
    if has_error_footnote:
        lines.append("")
        for agg in sorted_aggs:
            if agg.n_with_errors > 0:
                total_pairs = agg.n_clips + agg.n_with_errors
                lines.append(
                    f"*upstream prompt errors on " f"{agg.n_with_errors}/{total_pairs} pairs"
                )
                break
    lines.append("")

    # By Dimension
    lines.append("## By Dimension")
    lines.append("")
    dim_header_cells = [DIM_DISPLAY[d] for d in DIMENSIONS]
    lines.append("| Model | " + " | ".join(dim_header_cells) + " |")
    lines.append("|-------|" + "|".join(["------------:"] * len(DIMENSIONS)) + "|")
    for agg in sorted_aggs:
        suppressed = _suppressed_dim_means(agg, min_n=min_n)
        cells = [_fmt_float(suppressed.get(d)) for d in DIMENSIONS]
        lines.append(f"| {agg.model} | " + " | ".join(cells) + " |")
    lines.append("")

    # By Vertical
    lines.append("## By Vertical")
    lines.append("")
    lines.append("| Model | " + " | ".join(VERTICALS) + " |")
    lines.append("|-------|" + "|".join(["-------:"] * len(VERTICALS)) + "|")
    for agg in sorted_aggs:
        suppressed = _suppressed_vertical_means(agg, min_n=min_n)
        cells = [_fmt_float(suppressed.get(v)) for v in VERTICALS]
        lines.append(f"| {agg.model} | " + " | ".join(cells) + " |")
    lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "- Judge: Claude Opus 4.7 primary; GPT-5 swap for `claude-sonnet-4-6` outputs "
        "(no self-lineage grading per spec §13)."
    )
    lines.append(
        "- Composite weights: Accuracy 25 / OCR 10 / Spatial 15 / Targeted 15 / "
        "Hallucination 20 / Court-Grade 10 / Speed/Cost 5."
    )
    lines.append(
        "- Difficulty weights: easy=0.5, medium=1.0, hard=1.5 "
        "(override via `--difficulty-weights`)."
    )
    lines.append(
        "- Speed/Cost is mechanical (recomputed fresh from per-clip latency/cost data; "
        "free-tier always scores 10/10 on cost; paid models normalize against the paid "
        "lineup minimum)."
    )
    lines.append("- Per-clip detail: `reports/{model}.md` (local-only, gitignored).")
    lines.append("- Source DB: `outputs/scores.db` · Spec: `brief-eval-rig-spec-v0.2.md`.")
    lines.append("")
    lines.append("— end —")
    lines.append("")

    return "\n".join(lines)


# ---- Per-model report ----------------------------------------------------


def _rank_dimensions(
    dim_means: Mapping[str, float | None], *, top: bool
) -> list[tuple[str, float]]:
    """Top-3 (or bottom-3) dimensions by mean. None-valued dims are excluded.

    Ties broken by canonical ``DIMENSIONS`` order (declared in rubric).
    """
    canonical_index = {d: i for i, d in enumerate(DIMENSIONS)}
    candidates: list[tuple[str, float]] = [(d, v) for d, v in dim_means.items() if v is not None]
    if top:
        candidates.sort(key=lambda kv: (-kv[1], canonical_index[kv[0]]))
    else:
        candidates.sort(key=lambda kv: (kv[1], canonical_index[kv[0]]))
    return candidates[:3]


def render_per_model_report_md(agg: ModelAggregate) -> str:
    """Per-model deep-dive markdown report.

    Strengths/Weaknesses pull from non-None ``dim_means`` only. Per-Vertical
    table lists only verticals with ≥1 graded pair (sorted by composite desc).
    Per-Clip Detail is sorted by clip_id ascending.
    """
    lines: list[str] = []
    lines.append(f"# {agg.model}")
    lines.append("")
    lines.append(
        f"**Lineage:** {agg.lineage or EM_DASH} · "
        f"**Deployment:** {agg.deployment or EM_DASH} · "
        f"**n clips graded:** {agg.n_clips} · "
        f"**n with errors:** {agg.n_with_errors} · "
        f"**n spot-checked:** {agg.n_spot_checked}"
    )
    lines.append("")

    # Composite
    lines.append("## Composite")
    lines.append("")
    lines.append(f"- Overall: {_fmt_float(agg.composite_overall)}")
    lines.append(f"- Difficulty-adjusted: {_fmt_float(agg.composite_difficulty_adjusted)}")
    lines.append("")

    # Strengths
    lines.append("## Strengths (top 3 dimensions)")
    lines.append("")
    lines.append("| Dimension | Mean Score |")
    lines.append("|-----------|-----------:|")
    for dim, value in _rank_dimensions(agg.dim_means, top=True):
        lines.append(f"| {dim} | {_fmt_float(value)} |")
    lines.append("")

    # Weaknesses
    lines.append("## Weaknesses (bottom 3 dimensions)")
    lines.append("")
    lines.append("| Dimension | Mean Score |")
    lines.append("|-----------|-----------:|")
    for dim, value in _rank_dimensions(agg.dim_means, top=False):
        lines.append(f"| {dim} | {_fmt_float(value)} |")
    lines.append("")

    # Per-Vertical Means
    lines.append("## Per-Vertical Means")
    lines.append("")
    lines.append("| Vertical | n | Composite |")
    lines.append("|----------|--:|----------:|")
    vertical_counts: dict[str, int] = dict.fromkeys(VERTICALS, 0)
    for row in agg.per_clip:
        if row.has_judge_error:
            continue
        if row.vertical in vertical_counts:
            vertical_counts[row.vertical] += 1
    vertical_rows: list[tuple[str, int, float]] = []
    for vert in VERTICALS:
        mean = agg.vertical_means.get(vert)
        if mean is None or vertical_counts[vert] == 0:
            continue
        vertical_rows.append((vert, vertical_counts[vert], mean))
    vertical_rows.sort(key=lambda r: (-r[2], r[0]))
    for vert, n, mean in vertical_rows:
        lines.append(f"| {vert} | {n} | {_fmt_float(mean)} |")
    lines.append("")

    # Per-Clip Detail
    lines.append("## Per-Clip Detail")
    lines.append("")
    lines.append(
        "| Clip | Vertical | Difficulty | Composite | Latency total " "| Cost total | Error |"
    )
    lines.append(
        "|------|----------|------------|----------:|--------------:" "|-----------:|:-----:|"
    )
    for row in sorted(agg.per_clip, key=lambda r: r.clip_id):
        composite_cell = EM_DASH if row.has_judge_error else _fmt_float(row.composite)
        lines.append(
            f"| {row.clip_id} | {row.vertical or EM_DASH} | "
            f"{row.difficulty or EM_DASH} | {composite_cell} | "
            f"{_fmt_latency(float(row.latency_ms_total))} | "
            f"{_fmt_cost(row.cost_usd_total)} | "
            f"{'❌' if row.has_judge_error else ''} |"
        )
    lines.append("")

    return "\n".join(lines)


# ---- CSV -----------------------------------------------------------------


def _csv_value(value: float | int | str | None, *, is_float: bool) -> str:
    if value is None:
        return ""
    if is_float:
        return f"{float(value):.4f}"
    return str(value)


def render_metrics_csv(aggregates: list[ModelAggregate]) -> str:
    """Wide-format CSV. Column order is fixed via :data:`CSV_COLUMNS`.

    None values render as empty strings; floats render to 4 decimal places;
    rows sorted same as leaderboard.md (composite desc, model name asc).
    """
    sorted_aggs = _sorted_aggregates(aggregates)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for agg in sorted_aggs:
        row: dict[str, str] = {
            "model": _csv_value(agg.model, is_float=False),
            "lineage": _csv_value(agg.lineage, is_float=False),
            "deployment": _csv_value(agg.deployment, is_float=False),
            "n_clips": _csv_value(agg.n_clips, is_float=False),
            "n_with_errors": _csv_value(agg.n_with_errors, is_float=False),
            "n_spot_checked": _csv_value(agg.n_spot_checked, is_float=False),
            "composite_overall": _csv_value(agg.composite_overall, is_float=True),
            "composite_difficulty_adjusted": _csv_value(
                agg.composite_difficulty_adjusted, is_float=True
            ),
            "dim_description_accuracy_mean": _csv_value(
                agg.dim_means.get("description_accuracy"), is_float=True
            ),
            "dim_ocr_fidelity_mean": _csv_value(agg.dim_means.get("ocr_fidelity"), is_float=True),
            "dim_spatial_temporal_mean": _csv_value(
                agg.dim_means.get("spatial_temporal"), is_float=True
            ),
            "dim_targeted_query_mean": _csv_value(
                agg.dim_means.get("targeted_query"), is_float=True
            ),
            "dim_hallucination_resistance_mean": _csv_value(
                agg.dim_means.get("hallucination_resistance"), is_float=True
            ),
            "dim_court_grade_language_mean": _csv_value(
                agg.dim_means.get("court_grade_language"), is_float=True
            ),
            "dim_speed_cost_mean": _csv_value(agg.dim_means.get("speed_cost"), is_float=True),
            "vertical_retail_mean": _csv_value(agg.vertical_means.get("retail"), is_float=True),
            "vertical_commercial_mean": _csv_value(
                agg.vertical_means.get("commercial"), is_float=True
            ),
            "vertical_construction_mean": _csv_value(
                agg.vertical_means.get("construction"), is_float=True
            ),
            "vertical_schools_mean": _csv_value(agg.vertical_means.get("schools"), is_float=True),
            "vertical_industrial_mean": _csv_value(
                agg.vertical_means.get("industrial"), is_float=True
            ),
            "vertical_residential_mean": _csv_value(
                agg.vertical_means.get("residential"), is_float=True
            ),
            "vertical_edge_cases_mean": _csv_value(
                agg.vertical_means.get("edge_cases"), is_float=True
            ),
            "vertical_long_form_mean": _csv_value(
                agg.vertical_means.get("long_form"), is_float=True
            ),
            "latency_ms_p50": _csv_value(agg.latency_ms_p50, is_float=True),
            "cost_usd_total": _csv_value(agg.cost_usd_total, is_float=True),
        }
        writer.writerow(row)
    return buf.getvalue()


# ---- CLI -----------------------------------------------------------------


def _parse_difficulty_weights(raw: str) -> dict[str, float]:
    """Parse ``key=val,key=val,...`` into a dict. Reject unknown keys.

    Allowed keys mirror :data:`DEFAULT_DIFFICULTY_WEIGHTS`.
    """
    allowed = set(DEFAULT_DIFFICULTY_WEIGHTS.keys())
    out: dict[str, float] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise argparse.ArgumentTypeError(
                f"invalid difficulty weight {chunk!r}: expected key=value"
            )
        key, sep, val = chunk.partition("=")
        key = key.strip()
        val = val.strip()
        if key not in allowed:
            raise argparse.ArgumentTypeError(
                f"unknown difficulty key {key!r}: must be one of {sorted(allowed)}"
            )
        try:
            out[key] = float(val)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid difficulty weight value {val!r} for {key!r}"
            ) from exc
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m brief_eval_rig.reporting.leaderboard",
        description="Render leaderboard.md, per-model reports, and metrics.csv "
        "from a populated outputs/scores.db.",
    )
    parser.add_argument("--db", type=Path, default=Path("outputs/scores.db"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--corpus", type=Path, default=Path("corpus/"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs/"))
    parser.add_argument(
        "--difficulty-weights",
        type=_parse_difficulty_weights,
        default=None,
        help="Override difficulty weights, e.g. easy=0.5,medium=1.0,hard=1.5",
    )
    parser.add_argument("--min-n", type=int, default=1)
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Allow rendering on partial data (Phase 4 default; flag is informational).",
    )
    parser.add_argument("--kappa-avg", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    db_path: Path = args.db
    if not db_path.exists():
        print(f"error: db file not found at {db_path}", file=sys.stderr)
        return 1

    output_dir: Path = args.output_dir
    corpus_dir: Path = args.corpus
    outputs_dir: Path = args.outputs
    min_n: int = args.min_n
    kappa_avg: float | None = args.kappa_avg
    difficulty_weights: Mapping[str, float] = (
        args.difficulty_weights
        if args.difficulty_weights is not None
        else DEFAULT_DIFFICULTY_WEIGHTS
    )

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT COUNT(*) FROM judge_runs").fetchone()
        if int(row[0]) == 0:
            print(f"error: no judge_runs in {db_path}; nothing to report", file=sys.stderr)
            return 2

        clip_lookup = load_clip_lookup(corpus_dir)
        aggregates = aggregate_all_models(
            conn,
            clip_lookup=clip_lookup,
            outputs_dir=outputs_dir,
            difficulty_weights=difficulty_weights,
        )
    finally:
        conn.close()

    leaderboard_md = render_leaderboard_md(aggregates, kappa_average=kappa_avg, min_n=min_n)
    metrics_csv = render_metrics_csv(aggregates)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "leaderboard.md").write_text(leaderboard_md, encoding="utf-8")
    (output_dir / "metrics.csv").write_text(metrics_csv, encoding="utf-8")

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for agg in aggregates:
        per_model_md = render_per_model_report_md(agg)
        (reports_dir / f"{agg.model}.md").write_text(per_model_md, encoding="utf-8")

    kappa_str = _fmt_float(kappa_avg)
    print(
        f"wrote leaderboard.md ({len(aggregates)} models), "
        f"{len(aggregates)} per-model reports, "
        f"metrics.csv ({len(aggregates)} rows). κ avg: {kappa_str}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
