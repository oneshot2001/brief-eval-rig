"""Factory functions for the contender lineup.

Phase 1 ships the five cloud factories. Phase 2 adds three local Ollama-backed
factories. Each factory reads its credentials and model id at call time;
adapters are cheap to instantiate, so no caching.

Phase 2 deliberately drops Qwen3.5-VL 9B from the lineup — that tag is not
published on Ollama as of 2026-04-29. The Qwen lineage is still represented
locally by Qwen3.6 27B.
"""

from __future__ import annotations

from brief_eval_rig.adapters.base import VLMAdapter
from brief_eval_rig.adapters.claude import ClaudeAdapter
from brief_eval_rig.adapters.google import GoogleAdapter
from brief_eval_rig.adapters.ollama import OllamaAdapter
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


def qwen3_6_27b_local(base_url: str | None = None) -> OllamaAdapter:
    return OllamaAdapter(
        model_id="qwen3.6:27b",
        lineup_name="qwen3-6-27b-local",
        lineage="qwen",
        base_url=base_url,
    )


def nemotron_local(base_url: str | None = None) -> OllamaAdapter:
    # Reasoning model — same headroom rationale as the cloud variant.
    return OllamaAdapter(
        model_id="nemotron3",
        lineup_name="nemotron-3-nano-omni-local",
        lineage="nvidia",
        base_url=base_url,
        max_output_tokens=4096,
    )


def gemma4_26b_local(base_url: str | None = None) -> OllamaAdapter:
    return OllamaAdapter(
        model_id="gemma4:26b",
        lineup_name="gemma-4-26b-local",
        lineage="google",
        base_url=base_url,
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


def all_local_adapters(base_url: str | None = None) -> list[VLMAdapter]:
    """Return all three Phase 2 local contenders, freshly instantiated."""
    return [
        qwen3_6_27b_local(base_url),
        nemotron_local(base_url),
        gemma4_26b_local(base_url),
    ]


def all_adapters(ollama_base_url: str | None = None) -> list[VLMAdapter]:
    """All eight contenders — five cloud + three local — for Phase 5 use."""
    return all_cloud_adapters() + all_local_adapters(ollama_base_url)
