"""Per-million-token USD pricing constants for cloud contenders.

Verified against each provider's pricing page during Phase 1 pre-flight on
PRICING_AS_OF date below. When a provider changes prices, update the constant
*and* bump PRICING_AS_OF so downstream cost reports carry an audit trail.

Gemini 3.1 Pro Preview is tiered: <=200K input tokens uses the lower rate,
>200K uses the higher rate. The TIER1 constants are the lower rate; the rig
uses TIER1 by default since security clip prompts rarely exceed 200K tokens.
"""

from __future__ import annotations

from typing import NamedTuple

PRICING_AS_OF = "2026-04-29"


class TokenPrice(NamedTuple):
    """Per-million-token USD prices for input and output."""

    input_per_million: float
    output_per_million: float

    def cost_for(self, input_tokens: int, output_tokens: int) -> float:
        """Compute USD cost for a single call given response token counts."""
        return (input_tokens / 1_000_000.0) * self.input_per_million + (
            output_tokens / 1_000_000.0
        ) * self.output_per_million


# Anthropic — verified at https://platform.claude.com/docs/en/docs/about-claude/models
CLAUDE_SONNET_4_6 = TokenPrice(input_per_million=3.00, output_per_million=15.00)
CLAUDE_OPUS_4_7 = TokenPrice(input_per_million=5.00, output_per_million=25.00)

# Google AI Studio — verified at https://ai.google.dev/pricing
# Gemini 3.1 Pro Preview tier 1 (<=200K input tokens).
GEMINI_3_1_PRO_PREVIEW_TIER1 = TokenPrice(input_per_million=2.00, output_per_million=12.00)
GEMINI_3_1_PRO_PREVIEW_TIER2 = TokenPrice(input_per_million=4.00, output_per_million=18.00)
GEMINI_3_FLASH_PREVIEW = TokenPrice(input_per_million=0.50, output_per_million=3.00)

# OpenRouter — verified at https://openrouter.ai/api/v1/models
QWEN_3_6_FLASH = TokenPrice(input_per_million=0.25, output_per_million=1.50)
QWEN_3_6_27B = TokenPrice(input_per_million=0.325, output_per_million=3.25)
NEMOTRON_3_NANO_OMNI_FREE = TokenPrice(input_per_million=0.0, output_per_million=0.0)
