"""Sanity tests on the pricing constants."""

from __future__ import annotations

from brief_eval_rig.adapters.pricing import (
    CLAUDE_OPUS_4_7,
    CLAUDE_SONNET_4_6,
    GEMINI_3_1_PRO_PREVIEW_TIER1,
    GEMINI_3_FLASH_PREVIEW,
    NEMOTRON_3_NANO_OMNI_FREE,
    PRICING_AS_OF,
    QWEN_3_6_27B,
    QWEN_3_6_FLASH,
    TokenPrice,
)


def test_pricing_as_of_is_set() -> None:
    assert PRICING_AS_OF >= "2026-04-29"


def test_paid_models_have_nonzero_pricing() -> None:
    paid = [
        CLAUDE_SONNET_4_6,
        CLAUDE_OPUS_4_7,
        GEMINI_3_1_PRO_PREVIEW_TIER1,
        GEMINI_3_FLASH_PREVIEW,
        QWEN_3_6_FLASH,
        QWEN_3_6_27B,
    ]
    for p in paid:
        assert p.input_per_million > 0
        assert p.output_per_million > 0


def test_nemotron_free_tier_is_zero() -> None:
    assert NEMOTRON_3_NANO_OMNI_FREE.input_per_million == 0.0
    assert NEMOTRON_3_NANO_OMNI_FREE.output_per_million == 0.0


def test_cost_for_arithmetic() -> None:
    p = TokenPrice(input_per_million=2.0, output_per_million=10.0)
    # 1M input × $2 + 0.5M output × $10 = $2 + $5 = $7
    assert p.cost_for(1_000_000, 500_000) == 7.0


def test_output_cheaper_than_assumed_models() -> None:
    """Sanity: Sonnet output should be more expensive than Flash output."""
    assert CLAUDE_SONNET_4_6.output_per_million > GEMINI_3_FLASH_PREVIEW.output_per_million
