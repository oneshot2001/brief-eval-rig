"""Tests for scoring/rubric.py composite math and Speed/Cost normalization, plus judge loader."""

from __future__ import annotations

import math

import pytest

from brief_eval_rig.prompts.loader import load_judge_rubric
from brief_eval_rig.scoring.rubric import (
    DIMENSIONS,
    WEIGHTS,
    compute_composite,
    compute_speed_cost,
)


def test_weights_sum_to_one() -> None:
    assert math.isclose(sum(WEIGHTS.values()), 1.0, abs_tol=1e-9)


def test_dimensions_match_weights() -> None:
    assert set(DIMENSIONS) == set(WEIGHTS.keys())


def test_composite_all_tens() -> None:
    scores: dict[str, int | None] = {dim: 10 for dim in DIMENSIONS}
    assert compute_composite(scores) == pytest.approx(10.0)


def test_composite_all_zeros() -> None:
    scores: dict[str, int | None] = {dim: 0 for dim in DIMENSIONS}
    assert compute_composite(scores) == pytest.approx(0.0)


def test_composite_ocr_na_renormalizes() -> None:
    scores: dict[str, int | None] = {dim: 8 for dim in DIMENSIONS}
    scores["ocr_fidelity"] = None
    assert compute_composite(scores) == pytest.approx(8.0)


def test_composite_mixed_scores() -> None:
    scores: dict[str, int | None] = {
        "description_accuracy": 10,
        "ocr_fidelity": None,
        "spatial_temporal": 5,
        "targeted_query": 5,
        "hallucination_resistance": 10,
        "court_grade_language": 5,
        "speed_cost": 5,
    }
    weighted = 0.25 * 10 + 0.15 * 5 + 0.15 * 5 + 0.20 * 10 + 0.10 * 5 + 0.05 * 5
    weight_total = 0.25 + 0.15 + 0.15 + 0.20 + 0.10 + 0.05
    expected = weighted / weight_total
    assert compute_composite(scores) == pytest.approx(expected)


def test_composite_rejects_score_above_ten() -> None:
    scores: dict[str, int | None] = {dim: 5 for dim in DIMENSIONS}
    scores["description_accuracy"] = 11
    with pytest.raises(ValueError):
        compute_composite(scores)


def test_composite_rejects_negative_score() -> None:
    scores: dict[str, int | None] = {dim: 5 for dim in DIMENSIONS}
    scores["description_accuracy"] = -1
    with pytest.raises(ValueError):
        compute_composite(scores)


def test_composite_rejects_all_none() -> None:
    scores: dict[str, int | None] = {dim: None for dim in DIMENSIONS}
    with pytest.raises(ValueError):
        compute_composite(scores)


def test_speed_cost_at_lineup_minimum() -> None:
    assert compute_speed_cost(1000, 0.01, 1000, 0.01) == (10.0, 10.0)


def test_speed_cost_two_x_slower_and_costlier() -> None:
    latency, cost = compute_speed_cost(2000, 0.02, 1000, 0.01)
    assert latency == pytest.approx(5.0)
    assert cost == pytest.approx(5.0)


def test_speed_cost_four_x_slower() -> None:
    latency, _ = compute_speed_cost(4000, 0.01, 1000, 0.01)
    assert latency == pytest.approx(2.5)


def test_speed_cost_free_local_model_max_cost_score() -> None:
    _, cost = compute_speed_cost(1000, 0.0, 1000, 0.0)
    assert cost == 10.0


def test_speed_cost_free_model_in_paid_lineup_max_score() -> None:
    _, cost = compute_speed_cost(
        latency_ms=1000,
        cost_usd=0.0,
        lineup_min_latency_ms=1000,
        lineup_min_paid_cost_usd=0.001,
    )
    assert cost == 10.0


def test_speed_cost_all_free_lineup_returns_max_score() -> None:
    _, cost = compute_speed_cost(
        latency_ms=1000,
        cost_usd=0.0,
        lineup_min_latency_ms=1000,
        lineup_min_paid_cost_usd=0.0,
    )
    assert cost == 10.0


def test_speed_cost_paid_normalized_against_paid_min() -> None:
    _, cost = compute_speed_cost(
        latency_ms=1000,
        cost_usd=0.10,
        lineup_min_latency_ms=1000,
        lineup_min_paid_cost_usd=0.05,
    )
    assert cost == pytest.approx(5.0)


def test_speed_cost_paid_at_paid_minimum() -> None:
    _, cost = compute_speed_cost(
        latency_ms=1000,
        cost_usd=0.05,
        lineup_min_latency_ms=1000,
        lineup_min_paid_cost_usd=0.05,
    )
    assert cost == pytest.approx(10.0)


def test_speed_cost_paid_well_above_min() -> None:
    _, cost = compute_speed_cost(
        latency_ms=1000,
        cost_usd=1.00,
        lineup_min_latency_ms=1000,
        lineup_min_paid_cost_usd=0.05,
    )
    assert cost == pytest.approx(0.5)


def test_load_judge_rubric_contains_dimensions_and_schema() -> None:
    rendered = load_judge_rubric(
        ground_truth_summary="Subject in red hoodie enters at 0:04, exits at 0:23.",
        ground_truth_objects=["person", "shelf"],
        ground_truth_ocr=["EXIT", "AISLE 7"],
        expected_temporal_events=[
            {"time": "0:04", "event": "subject enters frame"},
            {"time": "0:23", "event": "subject exits frame"},
        ],
        targeted_query="Describe the person in red.",
        hallucination_trap="What does the license plate of the blue truck say?",
        generic_response="A person in red enters the aisle and walks toward the back.",
        targeted_response="The person in red appears to be a young adult wearing a hoodie.",
        hallucination_trap_response="I cannot determine that from this footage.",
        ocr_applicable="yes",
    )
    for dim in (
        "description_accuracy",
        "ocr_fidelity",
        "spatial_temporal",
        "targeted_query",
        "hallucination_resistance",
        "court_grade_language",
    ):
        assert dim in rendered
    assert '"scores"' in rendered
    assert '"justifications"' in rendered
    assert "Subject in red hoodie enters at 0:04, exits at 0:23." in rendered


def test_load_judge_rubric_ocr_not_applicable() -> None:
    rendered = load_judge_rubric(
        ground_truth_summary="Person walks across an empty parking lot at night.",
        ground_truth_objects=["person"],
        ground_truth_ocr=[],
        expected_temporal_events=[],
        targeted_query="What direction does the person walk?",
        hallucination_trap="What color is the vehicle parked in the lot?",
        generic_response="A person walks across the lot.",
        targeted_response="Left to right.",
        hallucination_trap_response="No vehicle is visible.",
        ocr_applicable="no",
    )
    assert "no" in rendered
    assert "N/A" in rendered
