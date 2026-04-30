# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2026-04-30

Patch — both Gemini contenders re-routed through OpenRouter (same model
weights, same prices) to sidestep Google AI Studio's free-tier rate limits
and the paid-tier gating on Gemini 3.1 Pro Preview. The lineup, lineage
labels, and per-clip JSON output names are unchanged; only the underlying
adapter changed from `GoogleAdapter` → `OpenRouterAdapter`.

### Changed

- `gemini_3_1_pro_preview()` and `gemini_3_flash_preview()` factories in
  `adapters/registry.py` now return `OpenRouterAdapter` instances pointing
  at `google/gemini-3.1-pro-preview` and `google/gemini-3-flash-preview`
  respectively. Pricing constants (`GEMINI_3_1_PRO_PREVIEW_TIER1`,
  `GEMINI_3_FLASH_PREVIEW`) are reused unchanged — OpenRouter mirrors
  Google's published rates.
- Gemini 3.1 Pro Preview gets `max_output_tokens=2048` (frontier reasoning-
  capable model wants headroom). Flash stays at the OpenRouter default 1024.

### Fixed

- Both Gemini adapters previously returned `400 INVALID_ARGUMENT — API key
  expired` on every call. Renewing the Google AI Studio key fixed Flash;
  Pro Preview remained blocked by free-tier quota (`429 RESOURCE_EXHAUSTED`).
  OpenRouter routing unblocks both without requiring a Google billing
  upgrade.

### Notes

- `adapters/google.py` is retained on disk (not deleted) so direct
  Google AI Studio routing can be revived if OpenRouter ever has an
  outage. It is no longer wired into the registry.
- `GOOGLE_AI_STUDIO_API_KEY` is no longer required for the cloud lineup.
  The `.env.example` entry stays as documentation; runtime no longer
  reads it unless someone manually instantiates `GoogleAdapter`.

## [0.4.0] - 2026-04-30

Phase 3 — LLM-as-judge + spot-check tooling + inter-rater agreement + SQLite
storage. The rig now grades any `outputs/{clip_id}/{model}.json` against
ground truth across all seven dimensions (0–10) with structured-output JSON,
mirrors results to SQLite, supports a 40-pair stratified human spot-check,
and reports Cohen's κ between judge and human.

### Added

- LLM-as-judge orchestrator (`scoring/llm_judge.py`, `scoring/judge.py`).
  Claude Opus 4.7 is the primary judge for seven of the eight contenders;
  GPT-5 swaps in for `claude-sonnet-4-6` outputs to avoid self-lineage
  grading (spec §13).
- Anthropic prompt caching engaged on the rubric block via
  `cache_control: {"type": "ephemeral"}`. Verified end-to-end through
  `cache_creation_input_tokens` / `cache_read_input_tokens` parsing in the
  live smoke tests.
- Spot-check CLI (`scoring/spot_check.py`, `scoring/spot_check_cli.py`)
  with stratified sampling — default 5 pairs per model × 8 models = 40 —
  plus `--sample` / `--resume` / `--status` modes and session persistence
  at `outputs/.spot_check_session.json`.
- Cohen's κ inter-rater agreement calculator (`scoring/agreement.py`)
  with a per-dimension Markdown report at `reports/inter_rater_agreement.md`.
- SQLite results storage at `outputs/scores.db` — three tables
  (`judge_runs`, `dimension_scores`, `spot_checks`) with foreign-key
  cascade and idempotent overwrite by `(clip_id, model)`.
- Rubric module (`scoring/rubric.py`) with the canonical seven-dimension
  weight vector and the mechanical Speed/Cost sub-score; canonical judge
  prompt template at `prompts/judge_rubric_template.txt` plus
  `prompts/loader.py::load_judge_rubric`.
- `runner/output_writer.py::update_with_judge` helper for in-place
  mutation of the score-related fields without disturbing prompts/metadata.
- `tests/live/test_smoke_judge.py` covering both judge routes
  (Anthropic → GPT-5 swap and non-Anthropic → Opus). Auto-skips on
  missing API keys or upstream prompt errors.

### Changed

- Spot-check stratification departs from the literal spec wording
  ("≥1 per `(model × vertical)` cell"). With 8 models × 9 verticals = 72
  cells and only 40 picks, the literal rule cannot fit — we stratify by
  model first (5 picks per model) then round-robin across the verticals
  each model has graded. Documented in
  `program-eval-phase3.md` Pre-flight #6.
- Cohen's κ implementation deviation: the spec called for
  `scipy.stats.cohen_kappa_score`, but that symbol lives in scikit-learn,
  not scipy. The κ formula was inlined in ~20 LOC using `numpy` directly,
  avoiding a heavyweight ML dependency for a single function.
- Pre-existing test hygiene: three mypy errors in `tests/conftest.py`
  (Any-return on JSON load; missing `cv2.VideoWriter_fourcc` stub) and
  `tests/test_registry.py` (unused `type: ignore`) cleaned up so
  `mypy src/ tests/` is fully green.

### Dependencies

- **Added:** `openai>=1.50.0`, `numpy>=1.26`.
- **Removed:** `scipy>=1.12`. It was specced for the κ calculator but the
  intended symbol is in sklearn, not scipy. Inlining κ in numpy makes
  scipy speculative; dropping it from the declared dependency list keeps
  the install footprint honest.

### Notes

- GPT-5 calls use `max_completion_tokens` (not `max_tokens`, which the
  newer chat-completions API rejects) plus `reasoning_effort="low"` so
  reasoning tokens don't starve the visible JSON output. Default
  `max_output_tokens` raised from 1500 → 4000 to give GPT-5 reasoning
  room while still being a hard cap on Anthropic's side. Both routes
  verified live against `outputs/smp-001/` model outputs.

## [0.3.1] - 2026-04-30

Phase 2 live-smoke verification. End-to-end run against a real Vast.ai A100
80GB instance (Montana host, $0.96/hr) confirmed all three local Ollama
adapters round-trip Q6325-LE video frames to GGUF Q4_K_M models and produce
non-error responses. Total smoke wall time 7m 06s.

### Changed

- **Nemotron tag corrected from `nemotron3` to `nemotron3:33b`** in the
  Ollama adapter registry. The bare `nemotron3` had no `:latest` alias on
  Ollama's registry; pulls failed with "manifest not found". The `:33b` tag
  resolves to `family: nemotron_h_omni`, the multimodal Nano Omni variant
  the spec wanted (33B params, 28 GB on disk).

### Added

- `scripts/smoke-local.sh` — paste-resistant wrapper for the local-lineup
  smoke run. Sidesteps multi-line argument mangling when invoking the
  orchestrator interactively.

### Notes

- Live-smoke output budget: ~$1 of Vast.ai compute end-to-end (instance
  launch through model pulls through smoke through destroy).
- **Known issue:** Gemma 4 26B generic-prompt response truncates at ~38
  chars mid-word ("...consists of a 15-"). Other two prompts return short
  but valid responses. Generation parameters (temperature, stop sequences,
  EOS handling) need investigation — not max_output_tokens, which defaults
  to 1024 and is far above the cutoff. Filed for a follow-up Vast session.
- **Known issue (carried from Phase 1):** Google AI Studio API key for the
  Gemini Project has expired. Both `gemini-3-1-pro-preview` and
  `gemini-3-flash-preview` return `API_KEY_INVALID`. Renew before Phase 5.

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
