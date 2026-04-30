# Brief Eval Rig

A reproducible evaluation framework that scores eight vision-language models on real
security video understanding tasks, producing a defensible public leaderboard.

> **One sentence.** A test rig that tells us — and the industry — which VLM is actually
> best for security video, on what dimensions, and at what cost.

## Status

| Phase | Description | Status |
|---|---|---|
| 0 | Bootstrap + foundation (scaffold, schema, loader) | done |
| 1 | Cloud adapters + frame sampler | done |
| 2 | Local adapters via Ollama | done |
| 3 | LLM-as-judge + spot-check tooling | done |
| 4 | Leaderboard generator | done |
| 5 | Full eval run + public report | pending |

Phase 2 ships the three local Ollama-backed adapters (Qwen3.6 27B, Nemotron 3 Nano
Omni, Gemma 4 26B A4B) plus a `--lineup {cloud,local,all}` orchestrator flag.
Combined with Phase 1, the rig now drives all eight contenders against a single
clip with one command.

## Quickstart

Requires Python 3.11+.

```bash
git clone https://github.com/oneshot2001/brief-eval-rig.git
cd brief-eval-rig
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                       # offline tests only
python -m brief_eval_rig.corpus.loader corpus/sample
```

To run the live smoke test against all five cloud providers, fill in `.env` from
`.env.example`, drop a short real `.mp4` at `corpus/sample/sample-001.mp4`, then:

```bash
pytest -m live
# or, equivalently, choose a lineup explicitly:
python -m brief_eval_rig.runner.orchestrator corpus/sample/sample-001.json --lineup cloud
python -m brief_eval_rig.runner.orchestrator corpus/sample/sample-001.json --lineup local
python -m brief_eval_rig.runner.orchestrator corpus/sample/sample-001.json --lineup all
```

For the `local` lineup, Ollama must be reachable. Default is `http://localhost:11434`;
override via `--ollama-url http://<host>:11434` or `OLLAMA_BASE_URL` (useful for
pointing at a remote Ollama instance such as Vast.ai). Models must already be pulled
on the target host (`ollama pull qwen3.6:27b`, etc.).

The orchestrator writes one JSON per adapter under `outputs/{clip_id}/{model}.json`
and prints a per-adapter latency / cost / error summary at the end.

### Grading model outputs (Phase 3)

Phase 3 fills the `scores` / `composite` / `judge` / `judge_justifications`
fields left null by Phase 1–2. Requires both `ANTHROPIC_API_KEY` and
`OPENAI_API_KEY` in `.env`. Results mirror to a SQLite database at
`outputs/scores.db` alongside the per-clip JSONs.

```bash
# Run the LLM judge against the existing smoke clip
python -m brief_eval_rig.scoring.judge outputs/smp-001/

# Walk a 40-pair stratified spot-check sample (interactive)
python -m brief_eval_rig.scoring.spot_check --sample 40

# Compute Cohen's κ inter-rater agreement (judge vs human)
python -m brief_eval_rig.scoring.agreement
```

### Generating the leaderboard (Phase 4)

After grading at least one (clip, model) pair via `python -m brief_eval_rig.scoring.judge`,
generate the canonical artifacts:

```bash
python -m brief_eval_rig.reporting.leaderboard
```

Writes three files:

- `leaderboard.md` at repo root — overall ranking + per-dimension + per-vertical sub-tables. Committed to git as the public canonical artifact (spec §12).
- `reports/{model}.md` for each model — strengths / weaknesses / per-vertical / per-clip detail. **Local-only (gitignored).**
- `metrics.csv` at repo root — wide-format, drops cleanly into Sheets / Notion / Vercel charts.

Flags:

| Flag | Default | Notes |
|------|---------|-------|
| `--db` | `outputs/scores.db` | Phase 3 SQLite output. |
| `--output-dir` | `.` | Where to write the three artifacts. |
| `--corpus` | `corpus/` | Clip metadata sidecars. |
| `--outputs` | `outputs/` | Per-clip JSON root (for fresh speed/cost calc). |
| `--min-n` | `1` | Suppress dim/vertical cells with fewer than N graded pairs. Pass `5` for Phase 5. |
| `--difficulty-weights` | `easy=0.5,medium=1.0,hard=1.5` | Override the difficulty-adjusted weights. |
| `--kappa-avg` | none | Cite a κ value in the leaderboard header. Compute via `python -m brief_eval_rig.scoring.agreement`. |

The composite formula and weights live in `scoring/rubric.py` (spec §6). Speed/cost is recomputed fresh from per-clip latency/cost data: free-tier (`cost_usd == 0.0`) always scores 10/10 on cost; paid models normalize against the paid lineup minimum.

## Contender lineup

Eight vision-language models across four lineages:

- **Anthropic.** Claude Sonnet 4.6.
- **Google.** Gemini 3.1 Pro Preview, Gemini 3 Flash Preview, Gemma 4 26B A4B (local).
- **Alibaba (Qwen).** Qwen3.6 Flash, Qwen3.6 27B (cloud + local).
- **NVIDIA.** Nemotron 3 Nano Omni (cloud + local).

Originally specced with Qwen3.5-VL 9B as a ninth entry; that tag is not published
on Ollama as of 2026-04-29, so the local Qwen lineup is represented by the 27B
variant alone.

## Methodology summary

Each contender is scored on seven dimensions (0–10) across a 50-clip corpus spanning
retail, commercial property, construction, schools, industrial, residential, and
edge-case verticals plus five long-form clips:

1. Description Accuracy (25%)
2. OCR Fidelity (10%)
3. Spatial / Temporal Reasoning (15%)
4. Targeted Query Handling (15%)
5. Hallucination Resistance (20%)
6. Court-Grade Language (10%)
7. Latency + Cost (5%)

Hallucination resistance carries the heaviest weight because legal-grade output is
the differentiator for the downstream use case.

Primary judge: Claude Opus 4.7. To eliminate self-preference bias, GPT-5 swaps in to
grade Claude Sonnet 4.6 outputs. A 10% stratified random sample is human-spot-checked
to track inter-rater agreement.

The full methodology document and leaderboard ship with the Phase 5 public report.

## Repository layout

```
brief-eval-rig/
├── corpus/                     # Clip JSON sidecars (videos are gitignored)
│   ├── retail/ commercial/ construction/ schools/ industrial/
│   ├── residential/ edge_cases/ long_form/
│   └── sample/                 # Sample clip for smoke tests
├── src/brief_eval_rig/
│   ├── adapters/               # VLMAdapter ABC, ClaudeAdapter, GoogleAdapter,
│   │                           # OpenRouterAdapter, registry, pricing
│   ├── corpus/                 # Pydantic schema + loader
│   ├── prompts/                # Three canonical prompt files + loader
│   ├── runner/                 # Orchestrator, frame sampler, output writer
│   ├── scoring/                # LLM judge + spot-check (Phase 3)
│   └── reporting/              # Leaderboard generator (Phase 4)
└── tests/
    ├── live/                   # Real-API smoke test, opt-in via -m live
    └── ...                     # Offline unit tests
```

## Hard rules

- No footage of identifiable minors, ever.
- No clip ships without documented license or consent.
- No API keys committed; `.env.example` shows placeholders only.
- Identical canonical prompts across all contenders. No per-model prompt tuning.
- Live tests require explicit `-m live` opt-in so default `pytest` never burns
  API credits.
- The leaderboard does not publish until inter-rater agreement (LLM judge vs.
  human spot-check) is documented.

## License

MIT. See [LICENSE](LICENSE).
