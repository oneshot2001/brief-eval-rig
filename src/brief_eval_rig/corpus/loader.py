"""Corpus loader and CLI entry point.

The loader is the single chokepoint for corpus admission. Every clip that
enters the rig passes through ``load_clip``, which validates the sidecar JSON
against ``ClipMetadata`` and surfaces any failure as a ``CorpusValidationError``.

Invoke as a CLI for a quick smoke test::

    python -m brief_eval_rig.corpus.loader corpus/sample
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import ValidationError

from brief_eval_rig.corpus.schema import ClipMetadata, CorpusValidationError


def load_clip(json_path: Path) -> ClipMetadata:
    """Load and validate a single clip metadata sidecar.

    Raises :class:`CorpusValidationError` on any schema, business-rule, or
    JSON-parse failure. The original Pydantic validation message is preserved
    in the exception chain via ``__cause__``.
    """
    if not json_path.is_file():
        raise CorpusValidationError(f"Clip sidecar not found: {json_path}")
    raw = json_path.read_text(encoding="utf-8")
    try:
        return ClipMetadata.model_validate_json(raw)
    except ValidationError as e:
        raise CorpusValidationError(f"Invalid clip metadata in {json_path}: {e}") from e
    except json.JSONDecodeError as e:
        raise CorpusValidationError(f"Malformed JSON in {json_path}: {e}") from e


def load_corpus(corpus_dir: Path) -> list[ClipMetadata]:
    """Recursively load every ``*.json`` sidecar under ``corpus_dir``.

    Skips ``.gitkeep`` and any non-``.json`` file. Returns clips in
    deterministic sorted-path order so eval runs are reproducible across
    machines.
    """
    if not corpus_dir.is_dir():
        return []
    paths = sorted(p for p in corpus_dir.rglob("*.json") if p.name != ".gitkeep")
    return [load_clip(p) for p in paths]


def list_corpus(corpus_dir: Path) -> None:
    """CLI: print loaded clips as a table.

    Invoked via ``python -m brief_eval_rig.corpus.loader <corpus_dir>``.
    """
    clips = load_corpus(corpus_dir)
    print(f"{len(clips)} clips loaded from {corpus_dir}")
    if not clips:
        return
    header = f"{'CLIP_ID':<12} {'VERTICAL':<14} {'DURATION':>10}  SOURCE"
    print(header)
    print("-" * len(header))
    for c in clips:
        print(f"{c.clip_id:<12} {c.vertical:<14} {c.duration_seconds:>9.1f}s  {c.source}")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(
            "Usage: python -m brief_eval_rig.corpus.loader <corpus_dir>",
            file=sys.stderr,
        )
        return 2
    list_corpus(Path(args[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
