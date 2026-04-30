"""Tests for LLMJudge: lineage routing, parsing, prompt-cache header, costs."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from brief_eval_rig.corpus.schema import ClipMetadata, TemporalEvent
from brief_eval_rig.scoring.llm_judge import LLMJudge
from brief_eval_rig.scoring.rubric import compute_composite

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _clip(*, ocr: list[str] | None = None) -> ClipMetadata:
    return ClipMetadata(
        clip_id="smp-001",
        vertical="retail",
        duration_seconds=10.0,
        resolution="1920x1080",
        framerate=30,
        scenario_tag="loitering",
        ground_truth_summary="Subject in red enters and exits.",
        ground_truth_objects=["person"],
        ground_truth_ocr=[] if ocr is None else ocr,
        expected_temporal_events=[
            TemporalEvent(time="0:04", event="subject enters frame"),
            TemporalEvent(time="0:23", event="subject exits frame"),
        ],
        targeted_query="Describe the person.",
        hallucination_trap="What does the license plate say?",
        difficulty="medium",
        source="synthetic",
        license="internal-test-only",
        minor_present=False,
        consent_documented=True,
    )


def _model_output(
    *,
    lineage: str = "qwen",
    model: str = "qwen3-6-flash",
    error_kinds: list[str] | None = None,
) -> dict[str, Any]:
    error_kinds = error_kinds or []
    prompts: dict[str, dict[str, Any]] = {}
    for kind in ("generic", "targeted", "hallucination_trap"):
        prompts[kind] = {
            "response": f"resp-{kind}",
            "latency_ms": 1000,
            "cost_usd": 0.01,
            "error": "boom" if kind in error_kinds else None,
        }
    return {
        "clip_id": "smp-001",
        "model": model,
        "lineage": lineage,
        "deployment": "cloud",
        "prompts": prompts,
        "scores": None,
    }


def _full_scores_payload(*, ocr_na: bool = False) -> dict[str, Any]:
    scores: dict[str, Any] = {
        "description_accuracy": 8,
        "ocr_fidelity": "N/A" if ocr_na else 7,
        "spatial_temporal": 6,
        "targeted_query": 9,
        "hallucination_resistance": 10,
        "court_grade_language": 7,
    }
    justifications = {
        "description_accuracy": "good summary",
        "ocr_fidelity": "no readable text" if ocr_na else "got most plates",
        "spatial_temporal": "ordering ok",
        "targeted_query": "direct answer",
        "hallucination_resistance": "refused cleanly",
        "court_grade_language": "mostly observational",
    }
    return {"scores": scores, "justifications": justifications}


def _anthropic_response(
    text: str,
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read: int = 0,
    cache_write: int = 0,
) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-opus-4-7",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_write,
            "cache_read_input_tokens": cache_read,
        },
    }


def _openai_response(
    text: str, *, prompt_tokens: int = 100, completion_tokens: int = 50
) -> dict[str, Any]:
    return {
        "id": "cmpl_1",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-5",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _make_judge() -> LLMJudge:
    return LLMJudge(anthropic_api_key="ak-test", openai_api_key="ok-test")


@respx.mock
def test_anthropic_lineage_routes_to_openai() -> None:
    judge = _make_judge()
    route = respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(200, json=_openai_response(json.dumps(_full_scores_payload())))
    )
    other = respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(500))
    run, scores = judge.judge_pair(
        clip_metadata=_clip(),
        model_output=_model_output(lineage="anthropic", model="claude-sonnet-4-6"),
        lineup_min_latency_ms=1000,
        lineup_min_cost_usd=0.03,
    )
    assert route.called
    assert not other.called
    assert run.error is None
    assert run.judge_model == "gpt-5"
    assert len(scores) == 7


@respx.mock
def test_non_anthropic_lineage_routes_to_anthropic() -> None:
    judge = _make_judge()
    route = respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            200, json=_anthropic_response(json.dumps(_full_scores_payload()))
        )
    )
    other = respx.post(OPENAI_URL).mock(return_value=httpx.Response(500))
    run, scores = judge.judge_pair(
        clip_metadata=_clip(ocr=["EXIT"]),
        model_output=_model_output(lineage="qwen", model="qwen3-6-flash"),
        lineup_min_latency_ms=1000,
        lineup_min_cost_usd=0.03,
    )
    assert route.called
    assert not other.called
    assert run.error is None
    assert run.judge_model == "claude-opus-4-7"
    assert len(scores) == 7


@respx.mock
def test_upstream_error_short_circuits() -> None:
    judge = _make_judge()
    a_route = respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(500))
    o_route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(500))
    run, scores = judge.judge_pair(
        clip_metadata=_clip(),
        model_output=_model_output(lineage="qwen", error_kinds=["generic"]),
        lineup_min_latency_ms=1000,
        lineup_min_cost_usd=0.03,
    )
    assert not a_route.called
    assert not o_route.called
    assert run.error is not None
    assert "upstream prompt errors" in run.error
    assert "generic" in run.error
    assert scores == []


@respx.mock
def test_malformed_json_response_records_error() -> None:
    judge = _make_judge()
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=_anthropic_response("not valid json {{{"))
    )
    run, scores = judge.judge_pair(
        clip_metadata=_clip(),
        model_output=_model_output(lineage="qwen"),
        lineup_min_latency_ms=1000,
        lineup_min_cost_usd=0.03,
    )
    assert run.error is not None
    assert "judge response parse failure" in run.error
    assert run.raw_judge_response is not None
    assert scores == []


@respx.mock
def test_missing_dimension_key_is_parse_failure() -> None:
    judge = _make_judge()
    payload = _full_scores_payload()
    del payload["scores"]["spatial_temporal"]
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=_anthropic_response(json.dumps(payload)))
    )
    run, _ = judge.judge_pair(
        clip_metadata=_clip(),
        model_output=_model_output(lineage="qwen"),
        lineup_min_latency_ms=1000,
        lineup_min_cost_usd=0.03,
    )
    assert run.error is not None
    assert "judge response parse failure" in run.error


@respx.mock
def test_extra_dimension_key_is_parse_failure() -> None:
    judge = _make_judge()
    payload = _full_scores_payload()
    payload["scores"]["bonus_dimension"] = 9
    payload["justifications"]["bonus_dimension"] = "extra"
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=_anthropic_response(json.dumps(payload)))
    )
    run, _ = judge.judge_pair(
        clip_metadata=_clip(),
        model_output=_model_output(lineage="qwen"),
        lineup_min_latency_ms=1000,
        lineup_min_cost_usd=0.03,
    )
    assert run.error is not None
    assert "judge response parse failure" in run.error


@respx.mock
def test_ocr_na_when_ground_truth_empty() -> None:
    judge = _make_judge()
    payload = _full_scores_payload(ocr_na=True)
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=_anthropic_response(json.dumps(payload)))
    )
    run, scores = judge.judge_pair(
        clip_metadata=_clip(ocr=[]),
        model_output=_model_output(lineage="qwen"),
        lineup_min_latency_ms=1000,
        lineup_min_cost_usd=0.03,
    )
    assert run.error is None
    ocr_score = next(s for s in scores if s.dimension == "ocr_fidelity")
    assert ocr_score.score is None
    assert ocr_score.justification is not None
    assert ocr_score.justification.startswith("N/A:")


@respx.mock
def test_anthropic_request_includes_cache_control() -> None:
    judge = _make_judge()
    captured: list[dict[str, Any]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_anthropic_response(json.dumps(_full_scores_payload())))

    respx.post(ANTHROPIC_URL).mock(side_effect=_handler)
    judge.judge_pair(
        clip_metadata=_clip(ocr=["EXIT"]),
        model_output=_model_output(lineage="qwen"),
        lineup_min_latency_ms=1000,
        lineup_min_cost_usd=0.03,
    )
    assert len(captured) == 1
    body = captured[0]
    block = body["messages"][0]["content"][0]
    assert block["type"] == "text"
    assert block["cache_control"] == {"type": "ephemeral"}


@respx.mock
def test_composite_matches_rubric() -> None:
    judge = _make_judge()
    payload = _full_scores_payload()
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=_anthropic_response(json.dumps(payload)))
    )
    run, scores = judge.judge_pair(
        clip_metadata=_clip(ocr=["EXIT"]),
        model_output=_model_output(lineage="qwen"),
        lineup_min_latency_ms=1000,
        lineup_min_cost_usd=0.03,
    )
    score_map: dict[str, int | None] = {s.dimension: s.score for s in scores}
    expected = compute_composite(score_map)
    assert run.composite == pytest.approx(expected)


@respx.mock
def test_speed_cost_dimension_always_populated() -> None:
    judge = _make_judge()
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            200, json=_anthropic_response(json.dumps(_full_scores_payload()))
        )
    )
    run, scores = judge.judge_pair(
        clip_metadata=_clip(ocr=["EXIT"]),
        model_output=_model_output(lineage="qwen"),
        lineup_min_latency_ms=500,
        lineup_min_cost_usd=0.015,
    )
    assert run.error is None
    sc = next(s for s in scores if s.dimension == "speed_cost")
    assert sc.score is not None
    assert 0 <= sc.score <= 10


@respx.mock
def test_anthropic_cost_uses_cached_price() -> None:
    judge = _make_judge()
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            200,
            json=_anthropic_response(
                json.dumps(_full_scores_payload()),
                input_tokens=200,
                output_tokens=100,
                cache_read=1000,
                cache_write=0,
            ),
        )
    )
    run, _ = judge.judge_pair(
        clip_metadata=_clip(ocr=["EXIT"]),
        model_output=_model_output(lineage="qwen"),
        lineup_min_latency_ms=1000,
        lineup_min_cost_usd=0.03,
    )
    assert run.error is None
    expected = 200 * (15.0 / 1_000_000) + 1000 * (1.50 / 1_000_000) + 100 * (75.0 / 1_000_000)
    assert run.judge_cost_usd == pytest.approx(expected)
    # Total input tokens include cached.
    assert run.judge_input_tokens == 1200
