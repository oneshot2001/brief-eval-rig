"""Factory functions for the five Phase 1 cloud contenders.

Each factory reads its API key (via the adapter constructor's load_dotenv),
its pricing constants, and its model id at call time. Adapters are cheap to
instantiate; no caching.
"""

from __future__ import annotations

from brief_eval_rig.adapters.base import VLMAdapter
from brief_eval_rig.adapters.claude import ClaudeAdapter
from brief_eval_rig.adapters.google import GoogleAdapter
from brief_eval_rig.adapters.openrouter import OpenRouterAdapter
from brief_eval_rig.adapters.pricing import (
    CLAUDE_SONNET_4_6,
    GEMINI_3_1_PRO_PREVIEW_TIER1,
    GEMINI_3_FLASH_PREVIEW,
    NEMOTRON_3_NANO_OMNI_FREE,
    QWEN_3_6_FLASH,
)


def claude_sonnet_4_6() -> ClaudeAdapter:
    return ClaudeAdapter(
        model_id="claude-sonnet-4-6",
        lineup_name="claude-sonnet-4-6",
        pricing=CLAUDE_SONNET_4_6,
    )


def gemini_3_1_pro_preview() -> GoogleAdapter:
    return GoogleAdapter(
        model_id="gemini-3.1-pro-preview",
        lineup_name="gemini-3-1-pro-preview",
        pricing=GEMINI_3_1_PRO_PREVIEW_TIER1,
    )


def gemini_3_flash_preview() -> GoogleAdapter:
    return GoogleAdapter(
        model_id="gemini-3-flash-preview",
        lineup_name="gemini-3-flash-preview",
        pricing=GEMINI_3_FLASH_PREVIEW,
    )


def qwen3_6_flash() -> OpenRouterAdapter:
    return OpenRouterAdapter(
        model_id="qwen/qwen3.6-flash",
        lineup_name="qwen3-6-flash",
        lineage="qwen",
        pricing=QWEN_3_6_FLASH,
        supports_native_video=True,
    )


def nemotron_cloud() -> OpenRouterAdapter:
    # Reasoning model: needs headroom for chain-of-thought + final answer.
    # Default 1024 left content=null after reasoning consumed all tokens.
    return OpenRouterAdapter(
        model_id="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        lineup_name="nemotron-3-nano-omni-cloud",
        lineage="nvidia",
        pricing=NEMOTRON_3_NANO_OMNI_FREE,
        supports_native_video=True,
        max_output_tokens=4096,
    )


def all_cloud_adapters() -> list[VLMAdapter]:
    """Return all five Phase 1 cloud contenders, freshly instantiated."""
    return [
        claude_sonnet_4_6(),
        gemini_3_1_pro_preview(),
        gemini_3_flash_preview(),
        qwen3_6_flash(),
        nemotron_cloud(),
    ]
