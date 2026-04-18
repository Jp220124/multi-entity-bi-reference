"""Typed models for events, classifications, analyses, and syntheses.

These Pydantic models are the contract at every LLM boundary. An LLM
response that cannot be parsed into the expected model is rejected and
never written to the context layer — this is what enables the "no black
box" guarantee: every downstream decision is based on validated, typed
inputs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class EntityType(str, Enum):
    """High-level business entity types in the reference portfolio.

    The real-world mapping is a client-specific taxonomy locked down
    during onboarding; these three are sufficient for the demo.
    """

    HOSPITALITY = "hospitality"
    RETAIL = "retail"
    REAL_ESTATE = "real_estate"


class EventCategory(str, Enum):
    """What the event is about, independent of which entity produced it."""

    SALE = "sale"
    EXPENSE = "expense"
    INVENTORY = "inventory"
    CUSTOMER = "customer"
    OPERATIONS = "operations"
    EXTERNAL_SIGNAL = "external_signal"


class EventPriority(str, Enum):
    """L1's routing decision.

    - ROUTINE: log and stop. No downstream analysis.
    - NOTABLE: route to L2 for cross-entity analysis.
    - URGENT:  skip L2 and route directly to L3 for executive synthesis.
    """

    ROUTINE = "routine"
    NOTABLE = "notable"
    URGENT = "urgent"


# ---------------------------------------------------------------------------
# Core records
# ---------------------------------------------------------------------------


class _TimestampedModel(BaseModel):
    """Base for records that need an ID and a creation timestamp."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Event(_TimestampedModel):
    """A raw business event entering the system.

    Examples:
        - A POS sale at a retail location.
        - A booking at a hotel.
        - A cost entry posted to the general ledger.
        - An external signal (cruise schedule, weather advisory).

    Upstream extractors are responsible for anonymizing the payload
    before constructing the Event. Personally identifiable information
    must never flow into the LLM tier.
    """

    source_system: str = Field(..., description="e.g. 'jewelry_pos', 'hotel_pms'")
    entity_hint: EntityType | None = Field(
        default=None,
        description="Optional hint from the extractor; L1 may override.",
    )
    payload: dict[str, Any] = Field(
        ...,
        description="Anonymized event fields. No PII.",
    )
    observed_at: datetime = Field(
        ...,
        description="When the event happened in the source system.",
    )


class Classification(_TimestampedModel):
    """L1's output: entity + category + priority + reasoning."""

    event_id: UUID
    entity: EntityType
    category: EventCategory
    priority: EventPriority
    rationale: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="One-sentence justification, human-readable.",
    )


class Analysis(_TimestampedModel):
    """L2's output: the cross-entity pattern, if any.

    `involves_entities` lists the entities the pattern spans. An
    analysis with a single entity is usually downgraded — cross-entity
    synthesis is the unique value of L2.
    """

    source_event_ids: list[UUID] = Field(..., min_length=1)
    involves_entities: list[EntityType] = Field(..., min_length=1)
    pattern_summary: str = Field(..., min_length=1, max_length=2000)
    recommended_action: str | None = Field(
        default=None,
        description=(
            "Optional concrete action a human operator could take. "
            "Omit if the analysis is purely informational."
        ),
    )
    escalate_to_l3: bool = Field(
        ...,
        description="L2's own recommendation on whether L3 should synthesize.",
    )


class Synthesis(_TimestampedModel):
    """L3's output: the portfolio-level briefing.

    The briefing is the most expensive artifact the system produces
    and is meant to be read by a human executive. It explicitly calls
    out contradictions and reversals — the things an L2-only system
    would miss.
    """

    analysis_ids: list[UUID] = Field(..., min_length=1)
    headline: str = Field(..., min_length=1, max_length=200)
    briefing: str = Field(..., min_length=1, max_length=4000)
    contradictions_detected: list[str] = Field(default_factory=list)
    suggested_next_watch: list[str] = Field(
        default_factory=list,
        description="Things the operator should keep an eye on in the next cycle.",
    )
