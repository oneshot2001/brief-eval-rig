"""Clip metadata schema and corpus loader.

Schema types are re-exported for convenience. Loader functions are NOT
re-exported here so that ``python -m brief_eval_rig.corpus.loader`` does not
trigger a runpy double-import warning.
"""

from brief_eval_rig.corpus.schema import (
    ClipMetadata,
    CorpusValidationError,
    Difficulty,
    TemporalEvent,
    Vertical,
)

__all__ = [
    "ClipMetadata",
    "CorpusValidationError",
    "Difficulty",
    "TemporalEvent",
    "Vertical",
]
