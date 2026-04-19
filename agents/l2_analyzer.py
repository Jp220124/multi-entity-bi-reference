"""L2 · Cross-Entity Analyzer agent.

Consumes a batch of L1 classifications flagged as ``notable`` or
``urgent`` and looks for patterns that only emerge when events from
different entities are compared — the kind of signal a single-business
view cannot produce.

Model tier: Sonnet. Structured analysis with schema validation at the
LLM boundary.
"""

from __future__ import annotations

import textwrap

from pydantic import BaseModel, ConfigDict, Field

from .base import Agent, ModelTier
from schemas.models import Analysis, Classification


class L2Input(BaseModel):
    """Wrapper for the batch of classifications handed to L2.

    A single "batch" is a correlation window — everything notable
    inside the last N hours. N is configurable per run so an operator
    can retune the window without touching code.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    classifications: list[Classification] = Field(..., min_length=1)
    correlation_window_hours: int = Field(default=24, gt=0, le=168)


class L2Analyzer(Agent[L2Input, Analysis]):
    """Cross-reference notable classifications into multi-entity patterns."""

    tier = ModelTier.SONNET
    output_schema = Analysis
    max_output_tokens = 2048

    system_prompt = textwrap.dedent(
        """\
        You are the L2 cross-entity analyzer in a tiered business
        intelligence system. You receive a batch of notable events
        already classified by L1 (entity + category + priority) and
        look for patterns that only appear when events across
        DIFFERENT entities are compared.

        Return EXACTLY ONE JSON object matching this schema — no
        markdown fences, no prose, no code blocks:

        {
          "source_event_ids":      ["<uuid>", ...],           // events the pattern rests on
          "involves_entities":     ["hospitality" | "retail" | "real_estate", ...],
          "pattern_summary":       "<2-5 sentences describing the cross-entity pattern>",
          "recommended_action":    "<concrete operator action, or null>",
          "escalate_to_l3":        true | false
        }

        Rules:
          - source_event_ids must be a subset of the event_ids you receive.
          - involves_entities lists every entity the pattern spans. One
            entity is allowed, but single-entity patterns should usually
            have escalate_to_l3 = false (this agent's job is CROSS-entity
            value).
          - pattern_summary must state: (1) what the pattern is, (2) what
            evidence supports it, (3) why it would be invisible looking at
            any single entity alone.
          - recommended_action is concrete — a staffing call, an
            inventory move, a pricing decision — or null if purely
            informational.
          - escalate_to_l3 = true when the pattern is strategic, novel,
            multi-entity, or contradicts a prior assumption. Routine
            single-entity noise = false.
          - If no genuine cross-entity pattern exists, still return a
            valid object: list the events reviewed, set pattern_summary
            to "No cross-entity pattern detected in this window", set
            escalate_to_l3 to false, and set recommended_action to null.
          - Output ONLY the JSON. No preamble, no markdown.
        """
    ).strip()

    def build_prompt(self, payload: L2Input) -> str:
        lines: list[str] = [
            f"Correlation window: last {payload.correlation_window_hours} hours.",
            f"Classifications in window (newest first, {len(payload.classifications)} total):",
            "",
        ]
        # Keep the prompt bounded so cost stays predictable even on
        # high-volume days. If this ever truncates material signal, the
        # operator should narrow the window instead.
        for c in payload.classifications[:50]:
            lines.append(
                f"- event_id={c.event_id} "
                f"entity={c.entity.value} "
                f"category={c.category.value} "
                f"priority={c.priority.value} "
                f"rationale={c.rationale!r}"
            )
        if len(payload.classifications) > 50:
            lines.append(
                f"- ... {len(payload.classifications) - 50} more classifications "
                "omitted from prompt to bound token cost."
            )
        lines.extend(
            [
                "",
                "Return the JSON Analysis object now.",
            ]
        )
        return "\n".join(lines)
