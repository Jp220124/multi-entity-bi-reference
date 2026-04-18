"""Schema validation tests.

The invariant under test: no downstream agent can ever consume an
LLM response that failed schema validation. Every boundary model
raises ``ValidationError`` on malformed input.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from schemas.models import (
    Analysis,
    Classification,
    EntityType,
    Event,
    EventCategory,
    EventPriority,
    Synthesis,
)


def test_event_rejects_extra_fields():
    with pytest.raises(ValidationError):
        Event(
            source_system="pos",
            payload={"a": 1},
            observed_at=datetime.now(timezone.utc),
            made_up_field="nope",  # type: ignore[call-arg]
        )


def test_classification_requires_nonempty_rationale():
    with pytest.raises(ValidationError):
        Classification(
            event_id=uuid4(),
            entity=EntityType.RETAIL,
            category=EventCategory.SALE,
            priority=EventPriority.ROUTINE,
            rationale="",
        )


def test_analysis_requires_at_least_one_event_and_entity():
    with pytest.raises(ValidationError):
        Analysis(
            source_event_ids=[],
            involves_entities=[],
            pattern_summary="No data, no pattern.",
            escalate_to_l3=False,
        )


def test_synthesis_rejects_empty_briefing():
    with pytest.raises(ValidationError):
        Synthesis(
            analysis_ids=[uuid4()],
            headline="Weekly Brief",
            briefing="",
        )


def test_classification_roundtrips_through_json():
    original = Classification(
        event_id=uuid4(),
        entity=EntityType.HOSPITALITY,
        category=EventCategory.OPERATIONS,
        priority=EventPriority.NOTABLE,
        rationale="Unusual late check-out pattern across multiple rooms.",
    )
    restored = Classification.model_validate_json(original.model_dump_json())
    assert restored == original
