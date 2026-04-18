"""L3 · Strategic Synthesis agent.

Consumes L2 analyses flagged for escalation and produces the
portfolio-level briefing a human executive actually reads. Uses
Claude Opus with extended thinking — the model is given a token
budget to deliberate before committing to an answer.

Status: scaffold. Prompt wiring lands in Sunday's pass.
"""

from __future__ import annotations

import os
import textwrap

from pydantic import BaseModel, ConfigDict, Field

from agents.base import Agent, ModelTier
from schemas.models import Analysis, Synthesis


class L3Input(BaseModel):
    """Batch of L2 analyses handed to L3 for synthesis."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    analyses: list[Analysis] = Field(..., min_length=1)


class L3Synthesizer(Agent[L3Input, Synthesis]):
    """Produce the portfolio-level briefing."""

    tier = ModelTier.OPUS
    output_schema = Synthesis
    max_output_tokens = 4096

    # Budget is sourced from the environment so it can be tuned without
    # code changes. Default is a balanced starting point; Week 3 of a
    # real engagement usually calibrates this against observed output
    # quality vs cost.
    extended_thinking_budget_tokens = int(
        os.environ.get("L3_THINKING_BUDGET_TOKENS", "8000")
    )

    system_prompt = textwrap.dedent(
        """\
        You are the L3 strategic synthesis agent. You receive a set of
        L2 analyses and produce one JSON Synthesis object containing
        a portfolio-level briefing for a human executive.

        Be explicit about contradictions, reversals, and anything the
        upstream L2 analyses may have missed when taken individually.
        """
    ).strip()

    def build_prompt(self, payload: L3Input) -> str:
        lines = ["Synthesize a portfolio briefing from these analyses:", ""]
        for a in payload.analyses[:20]:
            lines.append(
                f"- id={a.id} entities={[e.value for e in a.involves_entities]} "
                f"pattern={a.pattern_summary} "
                f"action={a.recommended_action or '(none)'}"
            )
        lines.append("")
        lines.append("Return the JSON Synthesis object now.")
        return "\n".join(lines)
