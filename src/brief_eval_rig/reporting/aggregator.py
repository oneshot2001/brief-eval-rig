"""Phase 4 aggregator: SQLite + per-clip JSON -> ModelAggregate dataclasses.

Pure read consumer of ``outputs/scores.db`` and the per-clip
``outputs/{clip_id}/{model}.json`` files written by Phase 3. Produces dataclass
aggregates that the renderer (Wave 2A) turns into Markdown / CSV.

Decoupled from ``scoring.storage`` deliberately — uses raw ``sqlite3`` queries
the same way ``scoring.agreement`` does, so the aggregator does not depend on
the storage helper schema.

Speed/cost handling: the v0.4.1 judge persisted a stale ``speed_cost`` value to
``dimension_scores`` before the rubric fix landed. Phase 4 ignores that column
entirely for ``speed_cost`` and recomputes the dimension fresh from raw
latency / cost data in the per-clip JSONs via
:func:`brief_eval_rig.scoring.rubric.compute_speed_cost`.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brief_eval_rig.corpus.loader import load_corpus
from brief_eval_rig.corpus.schema import ClipMetadata
from brief_eval_rig.scoring.rubric import (
    DIMENSIONS,
    compute_composite,
    compute_speed_cost,
)

DEFAULT_DIFFICULTY_WEIGHTS: dict[str, float] = {"easy": 0.5, "medium": 1.0, "hard": 1.5}

# Eight literal vertical values per ``corpus.schema.Vertical``. The ``sample``
# directory under ``corpus/`` is a fixture path, never a vertical value.
VERTICALS: tuple[str, ...] = (
    "retail",
    "commercial",
    "construction",
    "schools",
    "industrial",
    "residential",
    "edge_cases",
    "long_form",
)

# Six dimensions read from the DB; speed_cost is recomputed fresh.
_DB_DIMENSIONS: tuple[str, ...] = tuple(d for d in DIMENSIONS if d != "speed_cost")

_PROMPT_KEYS: tuple[str, ...] = ("generic", "targeted", "hallucination_trap")


@dataclass(frozen=True)
class PerClipRow:
    """One graded (clip, model) pair after fresh-speed_cost recomputation."""

    clip_id: str
    vertical: str
    difficulty: str
    composite: float
    latency_ms_total: int
    cost_usd_total: float
    has_judge_error: bool


@dataclass(frozen=True)
class ModelAggregate:
    """Aggregate metrics for one model across all its (clip, model) pairs."""

    model: str
    lineage: str
    deployment: str
    n_clips: int
    n_with_errors: int
    composite_overall: float
    composite_difficulty_adjusted: float
    dim_means: dict[str, float | None]
    vertical_means: dict[str, float | None]
    latency_ms_p50: float
    cost_usd_total: float
    n_spot_checked: int
    per_clip: tuple[PerClipRow, ...]


def load_clip_lookup(corpus_dir: Path) -> dict[str, ClipMetadata]:
    """Build a ``clip_id -> ClipMetadata`` index by walking the corpus directory."""
    return {clip.clip_id: clip for clip in load_corpus(corpus_dir)}


def load_per_clip_json(outputs_dir: Path, clip_id: str, model: str) -> dict[str, Any] | None:
    """Read ``outputs/{clip_id}/{model}.json``. Returns None when absent."""
    path = outputs_dir / clip_id / f"{model}.json"
    if not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8")
    parsed: dict[str, Any] = json.loads(raw)
    return parsed


def _per_prompt_metrics(payload: dict[str, Any]) -> tuple[list[int], list[float]]:
    """Extract per-prompt latency_ms and cost_usd lists from a per-clip JSON.

    Missing prompts contribute nothing (silently skipped). Non-numeric values
    are coerced via ``int`` / ``float`` so JSON-as-string ints still work.
    """
    prompts = payload.get("prompts", {})
    latencies: list[int] = []
    costs: list[float] = []
    for key in _PROMPT_KEYS:
        prompt = prompts.get(key)
        if not isinstance(prompt, dict):
            continue
        lat = prompt.get("latency_ms")
        if lat is not None:
            latencies.append(int(lat))
        cost = prompt.get("cost_usd")
        if cost is not None:
            costs.append(float(cost))
    return latencies, costs


def _all_models(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT DISTINCT model FROM judge_runs ORDER BY model").fetchall()
    return [row[0] for row in rows]


def _judge_runs_for_model(
    conn: sqlite3.Connection, model: str
) -> list[tuple[str, str, str | None]]:
    """Return ``(judge_run_id, clip_id, error)`` for one model's pairs."""
    rows = conn.execute(
        "SELECT judge_run_id, clip_id, error FROM judge_runs WHERE model = ? ORDER BY clip_id",
        (model,),
    ).fetchall()
    return [(row[0], row[1], row[2]) for row in rows]


def _dimension_scores_for_run(conn: sqlite3.Connection, judge_run_id: str) -> dict[str, int | None]:
    """Return ``dimension -> score`` for the six DB-sourced dimensions only."""
    rows = conn.execute(
        "SELECT dimension, score FROM dimension_scores WHERE judge_run_id = ?",
        (judge_run_id,),
    ).fetchall()
    out: dict[str, int | None] = {}
    for dim, score in rows:
        if dim in _DB_DIMENSIONS:
            out[dim] = None if score is None else int(score)
    return out


def _n_spot_checked(conn: sqlite3.Connection, model: str) -> int:
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT judge_run_id) FROM spot_checks
        WHERE judge_run_id IN (SELECT judge_run_id FROM judge_runs WHERE model = ?)
        """,
        (model,),
    ).fetchone()
    return int(row[0])


def _all_clip_lineup_metrics(
    conn: sqlite3.Connection,
    outputs_dir: Path,
) -> dict[str, tuple[int, float]]:
    """Compute per-clip lineup minima for latency and paid cost.

    For every distinct ``clip_id`` in ``judge_runs``, walk every model's
    per-clip JSON and compute:

    - ``lineup_min_latency_ms``: min of summed (generic+targeted+hallu_trap)
      per-prompt latency_ms across all models with that clip.
    - ``lineup_min_paid_cost_usd``: min summed cost across models whose
      summed cost is ``> 0`` (free-tier excluded). ``0.0`` when every model
      for that clip is free-tier.

    Pairs whose per-clip JSON is missing are skipped for the lineup math —
    they cannot contribute a latency / cost reading.
    """
    clip_models = conn.execute("SELECT DISTINCT clip_id, model FROM judge_runs").fetchall()
    by_clip: dict[str, list[tuple[int, float]]] = {}
    for clip_id, model in clip_models:
        payload = load_per_clip_json(outputs_dir, clip_id, model)
        if payload is None:
            continue
        latencies, costs = _per_prompt_metrics(payload)
        if not latencies and not costs:
            continue
        clip_lat_total = sum(latencies)
        clip_cost_total = sum(costs)
        by_clip.setdefault(clip_id, []).append((clip_lat_total, clip_cost_total))

    out: dict[str, tuple[int, float]] = {}
    for clip_id, totals in by_clip.items():
        min_lat = min(lat for lat, _ in totals)
        paid_costs = [cost for _, cost in totals if cost > 0.0]
        min_paid_cost = min(paid_costs) if paid_costs else 0.0
        out[clip_id] = (min_lat, min_paid_cost)
    return out


def aggregate_one_model(
    conn: sqlite3.Connection,
    model: str,
    *,
    clip_lookup: Mapping[str, ClipMetadata],
    outputs_dir: Path,
    difficulty_weights: Mapping[str, float] = DEFAULT_DIFFICULTY_WEIGHTS,
) -> ModelAggregate:
    """Aggregate one model's pairs into a :class:`ModelAggregate`.

    Pairs with a non-null ``judge_runs.error`` are excluded from
    ``composite_overall``, ``composite_difficulty_adjusted``, and the dim /
    vertical means, but counted in ``n_with_errors`` and surfaced in
    ``per_clip``.
    """
    runs = _judge_runs_for_model(conn, model)
    lineup_by_clip = _all_clip_lineup_metrics(conn, outputs_dir)

    lineage = ""
    deployment = ""
    composites: list[float] = []
    weighted_pairs: list[tuple[float, float]] = []  # (composite, weight)
    dim_score_lists: dict[str, list[float]] = {dim: [] for dim in DIMENSIONS}
    vertical_composites: dict[str, list[float]] = {v: [] for v in VERTICALS}
    all_latencies_per_prompt: list[int] = []
    all_costs_per_prompt: list[float] = []
    per_clip_rows: list[PerClipRow] = []
    n_with_errors = 0

    for judge_run_id, clip_id, error in runs:
        payload = load_per_clip_json(outputs_dir, clip_id, model)
        if payload is not None:
            if not lineage:
                lineage = str(payload.get("lineage", "") or "")
            if not deployment:
                deployment = str(payload.get("deployment", "") or "")
            latencies, costs = _per_prompt_metrics(payload)
        else:
            latencies, costs = [], []

        all_latencies_per_prompt.extend(latencies)
        all_costs_per_prompt.extend(costs)

        clip_latency_total = sum(latencies)
        clip_cost_total = sum(costs)
        has_error = error is not None

        clip_meta = clip_lookup.get(clip_id)
        vertical = clip_meta.vertical if clip_meta is not None else ""
        difficulty = clip_meta.difficulty if clip_meta is not None else ""

        if has_error:
            n_with_errors += 1
            # Surface error pair without a usable composite. Use 0.0 as a
            # placeholder for downstream rendering; the renderer should key off
            # has_judge_error to display the failure.
            per_clip_rows.append(
                PerClipRow(
                    clip_id=clip_id,
                    vertical=vertical,
                    difficulty=difficulty,
                    composite=0.0,
                    latency_ms_total=clip_latency_total,
                    cost_usd_total=clip_cost_total,
                    has_judge_error=True,
                )
            )
            continue

        db_scores = _dimension_scores_for_run(conn, judge_run_id)

        # Fresh speed_cost via the rubric helper.
        lineup_min_lat, lineup_min_paid_cost = lineup_by_clip.get(
            clip_id, (max(clip_latency_total, 1), 0.0)
        )
        latency_sub, cost_sub = compute_speed_cost(
            latency_ms=clip_latency_total,
            cost_usd=clip_cost_total,
            lineup_min_latency_ms=lineup_min_lat,
            lineup_min_paid_cost_usd=lineup_min_paid_cost,
        )
        speed_cost_score = (latency_sub + cost_sub) / 2.0

        scores_for_composite: dict[str, int | None] = dict(db_scores)
        scores_for_composite["speed_cost"] = int(round(speed_cost_score))
        composite = compute_composite(scores_for_composite)
        composites.append(composite)

        weight = float(difficulty_weights.get(difficulty, 1.0))
        weighted_pairs.append((composite, weight))

        # Per-dimension aggregation: DB dims keep their int / None contract,
        # speed_cost uses the float subscore average so the mean is unbiased.
        for dim in _DB_DIMENSIONS:
            score = db_scores.get(dim)
            if score is not None:
                dim_score_lists[dim].append(float(score))
        dim_score_lists["speed_cost"].append(speed_cost_score)

        if vertical in vertical_composites:
            vertical_composites[vertical].append(composite)

        per_clip_rows.append(
            PerClipRow(
                clip_id=clip_id,
                vertical=vertical,
                difficulty=difficulty,
                composite=composite,
                latency_ms_total=clip_latency_total,
                cost_usd_total=clip_cost_total,
                has_judge_error=False,
            )
        )

    n_clips = len(composites)
    composite_overall = (sum(composites) / n_clips) if n_clips > 0 else 0.0
    if weighted_pairs:
        weight_sum = sum(w for _, w in weighted_pairs)
        composite_difficulty_adjusted = (
            sum(c * w for c, w in weighted_pairs) / weight_sum if weight_sum > 0 else 0.0
        )
    else:
        composite_difficulty_adjusted = 0.0

    dim_means: dict[str, float | None] = {}
    for dim in DIMENSIONS:
        values = dim_score_lists[dim]
        dim_means[dim] = (sum(values) / len(values)) if values else None

    vertical_means: dict[str, float | None] = {}
    for vert in VERTICALS:
        values = vertical_composites[vert]
        vertical_means[vert] = (sum(values) / len(values)) if values else None

    latency_ms_p50 = (
        float(statistics.median(all_latencies_per_prompt)) if all_latencies_per_prompt else 0.0
    )
    cost_usd_total = sum(all_costs_per_prompt)

    n_spot_checked = _n_spot_checked(conn, model)

    return ModelAggregate(
        model=model,
        lineage=lineage,
        deployment=deployment,
        n_clips=n_clips,
        n_with_errors=n_with_errors,
        composite_overall=composite_overall,
        composite_difficulty_adjusted=composite_difficulty_adjusted,
        dim_means=dim_means,
        vertical_means=vertical_means,
        latency_ms_p50=latency_ms_p50,
        cost_usd_total=cost_usd_total,
        n_spot_checked=n_spot_checked,
        per_clip=tuple(per_clip_rows),
    )


def aggregate_all_models(
    conn: sqlite3.Connection,
    *,
    clip_lookup: Mapping[str, ClipMetadata],
    outputs_dir: Path,
    difficulty_weights: Mapping[str, float] = DEFAULT_DIFFICULTY_WEIGHTS,
) -> list[ModelAggregate]:
    """Aggregate every model present in ``judge_runs`` (alphabetical order)."""
    return [
        aggregate_one_model(
            conn,
            model,
            clip_lookup=clip_lookup,
            outputs_dir=outputs_dir,
            difficulty_weights=difficulty_weights,
        )
        for model in _all_models(conn)
    ]
