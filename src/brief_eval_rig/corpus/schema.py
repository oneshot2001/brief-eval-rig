"""Pydantic schema for clip metadata sidecars.

Every clip in ``corpus/`` ships with a JSON sidecar (e.g., ``ret-001.json``) that
this schema validates. Hard rules enforced at the schema layer:

- ``minor_present`` must be ``False``.
- ``consent_documented`` must be ``True``.
- ``clip_id`` must match ``[a-z]{2,3}-\\d{3,4}``.

Any sidecar violating these rules is rejected at load time. The intent is that
the loader is the single chokepoint for corpus admission — if the schema
accepts a clip, the rest of the rig can trust it.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CLIP_ID_PATTERN = re.compile(r"^[a-z]{2,3}-\d{3,4}$")

Vertical = Literal[
    "retail",
    "commercial",
    "construction",
    "schools",
    "industrial",
    "residential",
    "edge_cases",
    "long_form",
]

Difficulty = Literal["easy", "medium", "hard"]


class CorpusValidationError(ValueError):
    """Raised when a clip JSON fails schema or business-rule validation."""


class TemporalEvent(BaseModel):
    """One ground-truth temporal event within a clip."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    time: str
    event: str


class ClipMetadata(BaseModel):
    """Validated metadata sidecar for a single clip.

    Mirrors spec §5 exactly. Hard rules are enforced by ``field_validator`` so
    that downstream code can rely on every loaded clip having documented
    consent and no minors present.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    clip_id: str
    vertical: Vertical
    duration_seconds: float = Field(gt=0)
    resolution: str
    framerate: int = Field(gt=0)
    scenario_tag: str
    ground_truth_summary: str
    ground_truth_objects: list[str]
    ground_truth_ocr: list[str]
    expected_temporal_events: list[TemporalEvent]
    targeted_query: str
    hallucination_trap: str
    difficulty: Difficulty
    source: str
    license: str
    minor_present: bool
    consent_documented: bool

    @field_validator("clip_id")
    @classmethod
    def _validate_clip_id(cls, v: str) -> str:
        if not CLIP_ID_PATTERN.match(v):
            raise ValueError(
                f"clip_id {v!r} does not match required pattern '[a-z]{{2,3}}-\\d{{3,4}}'"
            )
        return v

    @field_validator("targeted_query", "hallucination_trap")
    @classmethod
    def _require_nonempty_query(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("targeted_query and hallucination_trap must be non-empty")
        return v

    @field_validator("minor_present")
    @classmethod
    def _reject_minors(cls, v: bool) -> bool:
        if v:
            raise ValueError(
                "minor_present must be False; clips featuring identifiable minors are not permitted"
            )
        return v

    @field_validator("consent_documented")
    @classmethod
    def _require_consent(cls, v: bool) -> bool:
        if not v:
            raise ValueError(
                "consent_documented must be True; undocumented clips cannot be ingested"
            )
        return v
