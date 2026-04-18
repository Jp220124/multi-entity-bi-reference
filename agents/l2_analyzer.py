"""L2 · Cross-Entity Analyzer agent.

Consumes L1 classifications flagged as ``notable`` or ``urgent`` and
looks for patterns that only emerge when events across different
entities are compared — the kind of signal a single-business view
cannot produce.

Model tier: Sonnet. Structured analysis with schema validation.

Status: scaffold. Prompt wiring lands in Sunday's pass.
"""

from __future__ import annotations

import textwrap

from pydantic import BaseModel, ConfigDict, Field

from agents.base import Agent, ModelTier
from schemas.models import Analysis, Classification


class L2Input(BaseModel):
    """Wrapper for the batch of classifications handed to L2."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    classifications: list[Classification] = Field(..., min_length=1)
    correlation_window_hours: int = 24


class L2Analyzer(Agent[L2Input, Analysis]):
    """Cross-reference classifications into multi-entity patterns."""

    tier = ModelTier.SONNET
    output_schema = Analysis
    max_output_tokens = 2048

    system_prompt = textwrap.dedent(
        """\
        You are the L2 cross-entity analyzer. You receive a batch of
        notable events already classified by L1 and look for patterns
        that only appear when events across different entities are
        compared.

        Produce one JSON object matching the Analysis schema. If no
        cross-entity pattern is evident, set escalate_to_l3 to false
        and summarise why in pattern_summary.
        """
    ).strip()

    def build_prompt(self, payload: L2Input) -> str:
        lines = [
            "Look for cross-entity patterns in this batch:",
            f"Correlation window: {payload.correlation_window_hours} hours.",
            "",
            "Classifications (most recent first):",
        ]
        for c in payload.classifications[:50]:
            lines.append(
                f"- id={c.id} event_id={c.event_id} entity={c.entity.value} "
                f"category={c.category.value} priority={c.priority.value} "
                f"rationale={c.rationale}"
            )
        lines.append("")
        lines.append("Return the JSON Analysis object now.")
        return "\n".join(lines)
