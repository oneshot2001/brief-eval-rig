"""Composite score math and Speed/Cost normalization for the seven scoring dimensions."""

from __future__ import annotations

from collections.abc import Mapping

DIMENSIONS: tuple[str, ...] = (
    "description_accuracy",
    "ocr_fidelity",
    "spatial_temporal",
    "targeted_query",
    "hallucination_resistance",
    "court_grade_language",
    "speed_cost",
)

WEIGHTS: dict[str, float] = {
    "description_accuracy": 0.25,
    "ocr_fidelity": 0.10,
    "spatial_temporal": 0.15,
    "targeted_query": 0.15,
    "hallucination_resistance": 0.20,
    "court_grade_language": 0.10,
    "speed_cost": 0.05,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "WEIGHTS must sum to 1.0"


def compute_composite(scores: Mapping[str, int | None]) -> float:
    weighted_sum = 0.0
    weight_total = 0.0
    any_present = False
    for dim, weight in WEIGHTS.items():
        score = scores.get(dim)
        if score is None:
            continue
        if score < 0 or score > 10:
            raise ValueError(f"Score for {dim!r} must be in 0..10, got {score}")
        weighted_sum += weight * score
        weight_total += weight
        any_present = True
    if not any_present:
        raise ValueError("compute_composite requires at least one non-None dimension score")
    return weighted_sum / weight_total


def compute_speed_cost(
    latency_ms: int,
    cost_usd: float,
    lineup_min_latency_ms: int,
    lineup_min_cost_usd: float,
) -> tuple[float, float]:
    if latency_ms == 0:
        latency_subscore = 10.0
    elif lineup_min_latency_ms == 0:
        latency_subscore = 0.0
    else:
        latency_subscore = max(0.0, min(10.0, 10.0 * lineup_min_latency_ms / latency_ms))

    if cost_usd == 0:
        cost_subscore = 10.0
    elif lineup_min_cost_usd == 0:
        cost_subscore = 0.0
    else:
        cost_subscore = max(0.0, min(10.0, 10.0 * lineup_min_cost_usd / cost_usd))

    return latency_subscore, cost_subscore
