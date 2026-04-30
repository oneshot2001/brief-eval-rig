# Brief Eval Rig — Leaderboard

Generated: 2026-04-30 · n_models: 8 · n_clips_max: 1 · κ averaged across dimensions: —

## Overall

| Rank | Model | Lineage | Deployment | n | Composite | Diff-Adj | Latency p50 | Cost total |
|-----:|-------|---------|------------|--:|----------:|---------:|------------:|-----------:|
| 1 | gemini-3-flash-preview | google | cloud | 1 | 6.22 | 6.22 | 7.42s | $0.0039 |
| 2 | qwen3-6-flash | qwen | cloud | 1 | 5.44 | 5.44 | 20.50s | $0.0184 |
| 3 | nemotron-3-nano-omni-local | nvidia | local | 1 | 4.33 | 4.33 | 26.52s | $0.0000 |
| 4 | qwen3-6-27b-local | qwen | local | 1 | 3.94 | 3.94 | 38.15s | $0.0000 |
| 5 | gemma-4-26b-local | google | local | 1 | 3.50 | 3.50 | 23.28s | $0.0000 |
| 6 | claude-sonnet-4-6 | anthropic | cloud | 1 | 3.44 | 3.44 | 15.54s | $0.1320 |
| 7 | gemini-3-1-pro-preview | google | cloud | 1 | 3.44 | 3.44 | 12.17s | $0.0406 |
| 8 | nemotron-3-nano-omni-cloud | nvidia | cloud | 1 | 2.33 | 2.33 | 18.28s | $0.0000 |

## By Dimension

| Model | Description | OCR | Spatial | Targeted | Hallucination | Court-Grade | Speed/Cost |
|-------|------------:|------------:|------------:|------------:|------------:|------------:|------------:|
| gemini-3-flash-preview | 2.00 | — | 3.00 | 9.00 | 10.00 | 8.00 | 10.00 |
| qwen3-6-flash | 2.00 | — | 3.00 | 8.00 | 9.00 | 8.00 | 2.93 |
| nemotron-3-nano-omni-local | 1.00 | — | 2.00 | 3.00 | 10.00 | 6.00 | 5.65 |
| qwen3-6-27b-local | 1.00 | — | 2.00 | 2.00 | 9.00 | 6.00 | 5.51 |
| gemma-4-26b-local | 1.00 | — | 0.00 | 0.00 | 10.00 | 6.00 | 6.35 |
| claude-sonnet-4-6 | 0.00 | — | 0.00 | 0.00 | 10.00 | 10.00 | 1.76 |
| gemini-3-1-pro-preview | 0.00 | — | 2.00 | 0.00 | 10.00 | 7.00 | 2.33 |
| nemotron-3-nano-omni-cloud | 0.00 | — | 1.00 | 0.00 | 6.00 | 4.00 | 6.71 |

## By Vertical

| Model | retail | commercial | construction | schools | industrial | residential | edge_cases | long_form |
|-------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| gemini-3-flash-preview | 6.22 | — | — | — | — | — | — | — |
| qwen3-6-flash | 5.44 | — | — | — | — | — | — | — |
| nemotron-3-nano-omni-local | 4.33 | — | — | — | — | — | — | — |
| qwen3-6-27b-local | 3.94 | — | — | — | — | — | — | — |
| gemma-4-26b-local | 3.50 | — | — | — | — | — | — | — |
| claude-sonnet-4-6 | 3.44 | — | — | — | — | — | — | — |
| gemini-3-1-pro-preview | 3.44 | — | — | — | — | — | — | — |
| nemotron-3-nano-omni-cloud | 2.33 | — | — | — | — | — | — | — |

## Methodology

- Judge: Claude Opus 4.7 primary; GPT-5 swap for `claude-sonnet-4-6` outputs (no self-lineage grading per spec §13).
- Composite weights: Accuracy 25 / OCR 10 / Spatial 15 / Targeted 15 / Hallucination 20 / Court-Grade 10 / Speed/Cost 5.
- Difficulty weights: easy=0.5, medium=1.0, hard=1.5 (override via `--difficulty-weights`).
- Speed/Cost is mechanical (recomputed fresh from per-clip latency/cost data; free-tier always scores 10/10 on cost; paid models normalize against the paid lineup minimum).
- Per-clip detail: `reports/{model}.md` (local-only, gitignored).
- Source DB: `outputs/scores.db` · Spec: `brief-eval-rig-spec-v0.2.md`.

— end —
