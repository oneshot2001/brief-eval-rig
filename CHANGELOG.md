# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
