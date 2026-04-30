"""LLM-as-judge orchestrator: routes lineage to Opus or GPT-5, parses, scores."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from typing import Any, Literal, cast

import anthropic
import httpx
import openai
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from brief_eval_rig.corpus.schema import ClipMetadata
from brief_eval_rig.prompts.loader import load_judge_rubric
from brief_eval_rig.scoring.rubric import compute_composite, compute_speed_cost
from brief_eval_rig.scoring.storage import DimensionScore, JudgeRun

_JUDGE_DIMENSIONS: tuple[str, ...] = (
    "description_accuracy",
    "ocr_fidelity",
    "spatial_temporal",
    "targeted_query",
    "hallucination_resistance",
    "court_grade_language",
)

_VALID_LINEAGES = {"qwen", "nvidia", "google", "anthropic"}

_SYSTEM_PROMPT = (
    "You are an expert evaluator scoring vision-language model outputs against "
    "ground truth. Return strict JSON only — no prose, no markdown fences."
)

_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-7": {
        "input": 15.0 / 1_000_000,
        "cache_read": 1.50 / 1_000_000,
        "cache_write": 18.75 / 1_000_000,
        "output": 75.0 / 1_000_000,
    },
    "gpt-5": {
        "input": 2.50 / 1_000_000,
        "output": 10.0 / 1_000_000,
    },
}


class _JudgeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: dict[str, int | Literal["N/A"]]
    justifications: dict[str, str]


class LLMJudge:
    def __init__(
        self,
        *,
        anthropic_model: str = "claude-opus-4-7",
        openai_model: str = "gpt-5",
        anthropic_api_key: str | None = None,
        openai_api_key: str | None = None,
        max_output_tokens: int = 4000,
        timeout_s: float = 120.0,
    ) -> None:
        load_dotenv()
        self._anthropic_model = anthropic_model
        self._openai_model = openai_model
        self._max_output_tokens = max_output_tokens
        self._timeout_s = timeout_s
        self._anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self._anthropic_client: Anthropic | None = None
        self._openai_client: OpenAI | None = None

    def _get_anthropic(self) -> Anthropic:
        if self._anthropic_client is None:
            if not self._anthropic_api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Add it to .env or the environment."
                )
            self._anthropic_client = Anthropic(api_key=self._anthropic_api_key)
        return self._anthropic_client

    def _get_openai(self) -> OpenAI:
        if self._openai_client is None:
            if not self._openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or the environment.")
            self._openai_client = OpenAI(api_key=self._openai_api_key)
        return self._openai_client

    def judge_pair(
        self,
        clip_metadata: ClipMetadata,
        model_output: dict[str, Any],
        lineup_min_latency_ms: int,
        lineup_min_cost_usd: float,
    ) -> tuple[JudgeRun, list[DimensionScore]]:
        run_id = uuid.uuid4().hex
        created_at = datetime.now(UTC).isoformat()
        clip_id = clip_metadata.clip_id
        model_name = str(model_output.get("model", ""))
        lineage = model_output.get("lineage")

        if lineage not in _VALID_LINEAGES:
            return (
                JudgeRun(
                    judge_run_id=run_id,
                    clip_id=clip_id,
                    model=model_name,
                    judge_model="",
                    composite=None,
                    error=f"unknown lineage: {lineage}",
                    created_at=created_at,
                ),
                [],
            )

        prompts = model_output.get("prompts") or {}
        upstream_errors = [
            kind
            for kind in ("generic", "targeted", "hallucination_trap")
            if (prompts.get(kind) or {}).get("error")
        ]
        judge_model = self._openai_model if lineage == "anthropic" else self._anthropic_model
        if upstream_errors:
            return (
                JudgeRun(
                    judge_run_id=run_id,
                    clip_id=clip_id,
                    model=model_name,
                    judge_model=judge_model,
                    composite=None,
                    error=f"upstream prompt errors: {upstream_errors}",
                    created_at=created_at,
                ),
                [],
            )

        ocr_applicable = "yes" if len(clip_metadata.ground_truth_ocr) > 0 else "no"
        rubric_prompt = load_judge_rubric(
            ground_truth_summary=clip_metadata.ground_truth_summary,
            ground_truth_objects=list(clip_metadata.ground_truth_objects),
            ground_truth_ocr=list(clip_metadata.ground_truth_ocr),
            expected_temporal_events=[
                e.model_dump() for e in clip_metadata.expected_temporal_events
            ],
            targeted_query=clip_metadata.targeted_query,
            hallucination_trap=clip_metadata.hallucination_trap,
            generic_response=str(prompts["generic"].get("response", "")),
            targeted_response=str(prompts["targeted"].get("response", "")),
            hallucination_trap_response=str(prompts["hallucination_trap"].get("response", "")),
            ocr_applicable=ocr_applicable,
        )

        try:
            if lineage == "anthropic":
                raw_text, raw_obj, usage_costs = self._call_openai(rubric_prompt)
            else:
                raw_text, raw_obj, usage_costs = self._call_anthropic(rubric_prompt)
        except Exception as e:
            return (
                JudgeRun(
                    judge_run_id=run_id,
                    clip_id=clip_id,
                    model=model_name,
                    judge_model=judge_model,
                    composite=None,
                    error=f"judge api failure: {type(e).__name__}: {e}",
                    created_at=created_at,
                ),
                [],
            )

        parsed_or_err = _parse_judge_response(raw_text, ocr_applicable)
        if isinstance(parsed_or_err, str):
            return (
                JudgeRun(
                    judge_run_id=run_id,
                    clip_id=clip_id,
                    model=model_name,
                    judge_model=judge_model,
                    composite=None,
                    error=f"judge response parse failure: {parsed_or_err}",
                    judge_input_tokens=usage_costs["input_tokens"],
                    judge_output_tokens=usage_costs["output_tokens"],
                    judge_cost_usd=usage_costs["cost_usd"],
                    raw_judge_response=json.dumps(raw_obj),
                    created_at=created_at,
                ),
                [],
            )

        scores_map, justifications_map = parsed_or_err

        latency_ms = int(prompts["generic"].get("latency_ms") or 0)
        cost_usd_total = sum(
            float(prompts[kind].get("cost_usd") or 0.0)
            for kind in ("generic", "targeted", "hallucination_trap")
        )
        latency_sub, cost_sub = compute_speed_cost(
            latency_ms=latency_ms,
            cost_usd=cost_usd_total,
            lineup_min_latency_ms=lineup_min_latency_ms,
            lineup_min_cost_usd=lineup_min_cost_usd,
        )
        speed_cost_score = round((latency_sub + cost_sub) / 2.0)

        dimension_scores: list[DimensionScore] = []
        composite_input: dict[str, int | None] = {}
        for dim in _JUDGE_DIMENSIONS:
            raw_score = scores_map[dim]
            justification = justifications_map[dim]
            if raw_score == "N/A":
                dimension_scores.append(
                    DimensionScore(
                        judge_run_id=run_id,
                        dimension=dim,  # type: ignore[arg-type]
                        score=None,
                        justification=f"N/A: {justification}",
                    )
                )
                composite_input[dim] = None
            else:
                int_score = int(raw_score)
                dimension_scores.append(
                    DimensionScore(
                        judge_run_id=run_id,
                        dimension=dim,  # type: ignore[arg-type]
                        score=int_score,
                        justification=justification,
                    )
                )
                composite_input[dim] = int_score

        speed_cost_justification = (
            f"latency_subscore={latency_sub:.2f}, cost_subscore={cost_sub:.2f} "
            f"(lineup_min_latency_ms={lineup_min_latency_ms}, "
            f"lineup_min_cost_usd={lineup_min_cost_usd:.6f})"
        )
        dimension_scores.append(
            DimensionScore(
                judge_run_id=run_id,
                dimension="speed_cost",
                score=speed_cost_score,
                justification=speed_cost_justification,
            )
        )
        composite_input["speed_cost"] = speed_cost_score

        composite = compute_composite(composite_input)

        return (
            JudgeRun(
                judge_run_id=run_id,
                clip_id=clip_id,
                model=model_name,
                judge_model=judge_model,
                composite=composite,
                error=None,
                judge_input_tokens=usage_costs["input_tokens"],
                judge_output_tokens=usage_costs["output_tokens"],
                judge_cost_usd=usage_costs["cost_usd"],
                raw_judge_response=json.dumps(raw_obj),
                created_at=created_at,
            ),
            dimension_scores,
        )

    @retry(
        retry=retry_if_exception_type(
            (httpx.NetworkError, httpx.TimeoutException, anthropic.APIConnectionError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _call_anthropic(self, rubric_prompt: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
        client = self._get_anthropic()
        messages = cast(
            "Any",
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": rubric_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        )
        response = client.messages.create(
            model=self._anthropic_model,
            max_tokens=self._max_output_tokens,
            system=_SYSTEM_PROMPT,
            messages=messages,
            timeout=self._timeout_s,
        )
        text_blocks = [b for b in response.content if getattr(b, "type", None) == "text"]
        raw_text = "".join(getattr(b, "text", "") for b in text_blocks)
        raw_obj: dict[str, Any] = response.model_dump()
        usage = response.usage
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        prices = _PRICING[self._anthropic_model]
        cost = (
            input_tokens * prices["input"]
            + cache_read * prices["cache_read"]
            + cache_write * prices["cache_write"]
            + output_tokens * prices["output"]
        )
        total_input = input_tokens + cache_read + cache_write
        return (
            raw_text,
            raw_obj,
            {
                "input_tokens": total_input,
                "output_tokens": output_tokens,
                "cost_usd": cost,
            },
        )

    @retry(
        retry=retry_if_exception_type(
            (httpx.NetworkError, httpx.TimeoutException, openai.APIConnectionError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _call_openai(self, rubric_prompt: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
        client = self._get_openai()
        response = client.chat.completions.create(
            model=self._openai_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": rubric_prompt},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=self._max_output_tokens,
            reasoning_effort="low",
            timeout=self._timeout_s,
        )
        choice = response.choices[0]
        raw_text = (choice.message.content or "").strip()
        raw_obj: dict[str, Any] = response.model_dump()
        usage = response.usage
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        prices = _PRICING[self._openai_model]
        cost = input_tokens * prices["input"] + output_tokens * prices["output"]
        return (
            raw_text,
            raw_obj,
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
            },
        )


def _parse_judge_response(
    raw_text: str, ocr_applicable: str
) -> tuple[dict[str, int | str], dict[str, str]] | str:
    if not raw_text.strip():
        return "empty response"
    try:
        obj = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return f"invalid json: {e}"
    try:
        parsed = _JudgeResponse.model_validate(obj)
    except ValidationError as e:
        return f"schema mismatch: {e}"

    expected = set(_JUDGE_DIMENSIONS)
    score_keys = set(parsed.scores.keys())
    just_keys = set(parsed.justifications.keys())
    if score_keys != expected:
        missing = expected - score_keys
        extra = score_keys - expected
        return f"scores keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
    if just_keys != expected:
        missing = expected - just_keys
        extra = just_keys - expected
        return f"justifications keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"

    for dim, val in parsed.scores.items():
        if val == "N/A":
            if dim != "ocr_fidelity":
                return f"N/A only allowed for ocr_fidelity, got it on {dim!r}"
            if ocr_applicable == "yes":
                return "judge returned N/A for ocr_fidelity but ocr_applicable=yes"
        else:
            if not isinstance(val, int) or val < 0 or val > 10:
                return f"score for {dim!r} not an int in 0..10: {val!r}"

    return cast("dict[str, int | str]", dict(parsed.scores)), dict(parsed.justifications)
