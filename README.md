# Brief Eval Rig

A reproducible evaluation framework that scores nine vision-language models on real
security video understanding tasks, producing a defensible public leaderboard.

> **One sentence.** A test rig that tells us — and the industry — which VLM is actually
> best for security video, on what dimensions, and at what cost.

## Status

| Phase | Description | Status |
|---|---|---|
| 0 | Bootstrap + foundation (scaffold, schema, loader) | done |
| 1 | Cloud adapters + frame sampler | pending |
| 2 | Local adapters via Ollama | pending |
| 3 | LLM-as-judge + spot-check tooling | pending |
| 4 | Leaderboard generator | pending |
| 5 | Full eval run + public report | pending |

Phase 0 ships the abstractions only. Concrete model adapters land in Phases 1 and 2.

## Quickstart

Requires Python 3.11+.

```bash
git clone https://github.com/oneshot2001/brief-eval-rig.git
cd brief-eval-rig
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m brief_eval_rig.corpus.loader corpus/sample
```

The last command should print one row for clip `smp-001` from the sample corpus.

## Contender lineup

Nine vision-language models across four lineages:

- **Anthropic.** Claude Sonnet 4.7.
- **Google.** Gemini 3.1 Pro Preview, Gemini 3 Flash Preview, Gemma 4 26B A4B (local).
- **Alibaba (Qwen).** Qwen3.6 Flash, Qwen3.6 27B (cloud + local), Qwen3.5-VL 9B (local).
- **NVIDIA.** Nemotron 3 Nano Omni (cloud + local).

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
grade Claude Sonnet 4.7 outputs. A 10% stratified random sample is human-spot-checked
to track inter-rater agreement.

The full methodology document and leaderboard ship with the Phase 5 public report.

## Repository layout

```
brief-eval-rig/
├── corpus/                  # Clip JSON sidecars (videos are gitignored)
│   ├── retail/
│   ├── commercial/
│   ├── construction/
│   ├── schools/
│   ├── industrial/
│   ├── residential/
│   ├── edge_cases/
│   ├── long_form/
│   └── sample/              # Sample clip for smoke tests
├── src/brief_eval_rig/
│   ├── adapters/            # VLMAdapter ABC; concrete adapters land in Phase 1+
│   ├── corpus/              # Pydantic schema + loader
│   ├── prompts/             # Canonical prompts (Phase 1)
│   ├── runner/              # Orchestrator + frame sampler (Phase 1)
│   ├── scoring/             # LLM judge + spot-check (Phase 3)
│   └── reporting/           # Leaderboard generator (Phase 4)
└── tests/
```

## Hard rules

- No footage of identifiable minors, ever.
- No clip ships without documented license or consent.
- No API keys committed; `.env.example` shows placeholders only.
- Identical canonical prompts across all contenders. No per-model prompt tuning.
- The leaderboard does not publish until inter-rater agreement (LLM judge vs.
  human spot-check) is documented.

## License

MIT. See [LICENSE](LICENSE).
