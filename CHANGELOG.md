# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-29

Phase 2 — Local adapters via Ollama. Three local contenders (Qwen3.6 27B,
Nemotron 3 Nano Omni, Gemma 4 26B A4B) now drive against the same one-clip
smoke orchestrator as the cloud lineup, with the same spec-§9 output schema.

### Added

- `OllamaAdapter` (single class, three configurations via factory functions).
  Frame-samples using Phase 1's sampler since Ollama's `/api/chat` API
  accepts only images, not video. Reasoning fallback (`message.thinking`)
  for Nemotron parallel to the OpenRouter cloud variant.
- Three local-adapter factory functions plus `all_local_adapters()` and
  `all_adapters()` registry helpers. Local factories accept an optional
  `base_url` to point at a remote Ollama (Vast.ai or other).
- Orchestrator CLI gained `--lineup {cloud,local,all}` and `--ollama-url`
  flags. Default lineup remains `cloud` (Phase 1 behavior preserved).
- Live integration test for the local lineup at `tests/live/test_smoke_local.py`,
  marked `@pytest.mark.live`. Auto-skips if Ollama is unreachable.

### Changed

- Lineup contracted from nine to eight contenders. **Qwen3.5-VL 9B dropped**:
  no Ollama tag is published for it as of 2026-04-29. Documented in README.

### Notes

- Local adapters cost `$0.00` per call by design. Vast.ai compute is amortized
  at the operational layer rather than recorded in `AnalysisResult.cost_usd`.
- Ollama does not require authentication; no API-key handling was added.
- Vast.ai launch remains a manual operational step. Adapter code is location-
  agnostic — same class drives `localhost:11434` or a remote IP.

## [0.2.0] - 2026-04-29

Phase 1 — Cloud adapters + frame sampler. Five cloud contenders can now answer
questions about a real clip end-to-end, with cost and latency tracking, written
to per-clip per-model JSON outputs ready for Phase 3 scoring.

### Added

- Three adapter implementations covering five cloud contenders:
  - `ClaudeAdapter` (Claude Sonnet 4.6 via Anthropic Messages API; frame-samples).
  - `GoogleAdapter` (Gemini 3.1 Pro Preview + Gemini 3 Flash Preview via
    google-genai Files API; native video).
  - `OpenRouterAdapter` (Qwen3.6 Flash + Nemotron 3 Nano Omni cloud via OpenRouter
    OpenAI-compatible chat completions; data-URI video).
- Adapter registry / factory functions in `adapters/registry.py`.
- Per-million-token pricing constants verified against provider docs on 2026-04-29
  in `adapters/pricing.py`, with a `cost_for(input, output)` helper.
- Deterministic frame sampler (`runner/frame_sampler.py`) using OpenCV with
  uniform-interval extraction plus optional histogram-diff scene-change keyframes.
- Canonical prompt set: generic brief, targeted query template, hallucination trap
  template under `src/brief_eval_rig/prompts/`.
- One-clip smoke-test orchestrator (`runner/orchestrator.py`) and JSON output writer
  (`runner/output_writer.py`) matching spec §9 schema exactly.
- pytest live-test marker (`-m live`); default invocation skips network tests.

### Changed

- Clip metadata schema now requires `targeted_query` and `hallucination_trap`
  string fields (used by Prompts B and C). Both must be non-empty.
- Sample clip (`corpus/sample/sample-001.json`) updated with placeholder values
  for the two new fields.
- Anthropic Sonnet contender corrected from "Sonnet 4.7" (does not exist) to
  Sonnet 4.6 (`claude-sonnet-4-6`). Judge model (Opus 4.7) unchanged.

### Notes

- OpenRouter video request format uses `image_url` content blocks with
  `data:video/mp4;base64,...` data URIs. This is the broadly-supported
  OpenAI-compatible pattern; if a specific model rejects it, a frame-sampled
  fallback will be added in a follow-up.
- Nemotron 3 Nano Omni is on OpenRouter's `:free` tier (zero per-token cost,
  daily and per-minute rate limits). Acceptable for Phase 1 / 2 dev cadence;
  Vast.ai local fallback exists for Phase 5 if rate-limited.

## [0.1.0] - 2026-04-28

Phase 0 bootstrap. Foundation only — no working adapters yet.

### Added

- Repository scaffolding: `pyproject.toml`, `LICENSE` (MIT), `.gitignore`, `.env.example`.
- Base abstractions: `VLMAdapter` ABC, `Clip` and `AnalysisResult` value types,
  `Lineage` and `Deployment` literal types.
- Clip metadata schema (Pydantic v2) with hard rules: minors must be absent,
  consent must be documented, clip IDs must match `[a-z]{2,3}-\d{3,4}`.
- Corpus loader and CLI entry point: `python -m brief_eval_rig.corpus.loader <dir>`.
- Sample clip (`corpus/sample/sample-001.json`) for smoke testing.
- pytest test suite covering schema validation and corpus loading.
