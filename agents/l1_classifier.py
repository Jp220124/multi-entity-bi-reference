"""L1 · Classification & Routing agent.

Runs on every raw event entering the system. Decides which entity the
event belongs to, what category it falls into, and whether it is
routine (stop here), notable (send to L2), or urgent (send straight to
L3).

Model tier: Haiku. High volume, low latency, cheap per call.
"""

from __future__ import annotations

import textwrap

from agents.base import Agent, ModelTier
from schemas.models import Classification, Event


class L1Classifier(Agent[Event, Classification]):
    """Classify + route a single raw event."""

    tier = ModelTier.HAIKU
    output_schema = Classification
    max_output_tokens = 512

    system_prompt = textwrap.dedent(
        """\
        You are the L1 triage agent in a tiered business-intelligence
        system. For every raw event you receive, output one strict JSON
        object matching this schema — no prose, no code fences:

        {
          "event_id":  "<uuid of the source event>",
          "entity":    "hospitality" | "retail" | "real_estate",
          "category":  "sale" | "expense" | "inventory" |
                       "customer" | "operations" | "external_signal",
          "priority":  "routine" | "notable" | "urgent",
          "rationale": "<one short sentence, under 160 characters>"
        }

        Rules:
          - Choose exactly one entity, category, and priority value.
          - Use the entity_hint if the event provides one and it still
            makes sense for the payload.
          - routine  = expected activity, no deeper analysis needed.
          - notable  = may be part of a cross-entity pattern; send up.
          - urgent   = operationally time-sensitive, skip L2.
          - rationale explains the priority call in plain language.
          - Output ONLY the JSON object. No markdown, no prose.
        """
    ).strip()

    def build_prompt(self, payload: Event) -> str:
        entity_hint_line = (
            f"- entity_hint: {payload.entity_hint.value}"
            if payload.entity_hint
            else "- entity_hint: (none)"
        )
        return textwrap.dedent(
            f"""\
            Classify this event.

            - event_id: {payload.id}
            - source_system: {payload.source_system}
            {entity_hint_line}
            - observed_at: {payload.observed_at.isoformat()}
            - payload: {payload.payload}

            Return the JSON object now.
            """
        ).strip()
