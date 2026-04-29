"""Schema-level tests for ClipMetadata."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from brief_eval_rig.corpus.schema import ClipMetadata


def test_valid_payload_loads(valid_payload: dict[str, Any]) -> None:
    clip = ClipMetadata.model_validate(valid_payload)
    assert clip.clip_id == "smp-001"
    assert clip.vertical == "retail"
    assert clip.expected_temporal_events[0].time == "0:04"


def test_missing_required_field_rejected(valid_payload: dict[str, Any]) -> None:
    valid_payload.pop("vertical")
    with pytest.raises(ValidationError):
        ClipMetadata.model_validate(valid_payload)


def test_consent_undocumented_rejected(valid_payload: dict[str, Any]) -> None:
    valid_payload["consent_documented"] = False
    with pytest.raises(ValidationError):
        ClipMetadata.model_validate(valid_payload)


def test_minor_present_rejected(valid_payload: dict[str, Any]) -> None:
    valid_payload["minor_present"] = True
    with pytest.raises(ValidationError):
        ClipMetadata.model_validate(valid_payload)


def test_invalid_clip_id_rejected(valid_payload: dict[str, Any]) -> None:
    valid_payload["clip_id"] = "BadID-1"
    with pytest.raises(ValidationError):
        ClipMetadata.model_validate(valid_payload)


def test_invalid_vertical_rejected(valid_payload: dict[str, Any]) -> None:
    valid_payload["vertical"] = "aerospace"
    with pytest.raises(ValidationError):
        ClipMetadata.model_validate(valid_payload)


def test_targeted_query_present(valid_payload: dict[str, Any]) -> None:
    clip = ClipMetadata.model_validate(valid_payload)
    assert clip.targeted_query.strip() != ""


def test_empty_targeted_query_rejected(valid_payload: dict[str, Any]) -> None:
    valid_payload["targeted_query"] = "   "
    with pytest.raises(ValidationError):
        ClipMetadata.model_validate(valid_payload)


def test_empty_hallucination_trap_rejected(valid_payload: dict[str, Any]) -> None:
    valid_payload["hallucination_trap"] = ""
    with pytest.raises(ValidationError):
        ClipMetadata.model_validate(valid_payload)
