"""L3 · Strategic Synthesis agent.

Consumes the L2 analyses that were flagged for escalation and produces
the portfolio-level briefing a human executive actually reads. Uses
Claude Opus with extended thinking — the model is given a bounded
token budget to deliberate before committing to an answer.

L3 is the most expensive tier per call. Keep the batch size small.
"""

from __future__ import annotations

import os
import textwrap

from pydantic import BaseModel, ConfigDict, Field

from schemas.models import Analysis, Synthesis

from .base import Agent, ModelTier


class L3Input(BaseModel):
    """Batch of L2 analyses handed to L3 for synthesis."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    analyses: list[Analysis] = Field(..., min_length=1)


class L3Synthesizer(Agent[L3Input, Synthesis]):
    """Produce the portfolio-level briefing from escalated analyses."""

    tier = ModelTier.OPUS
    output_schema = Synthesis
    max_output_tokens = 4096

    # Thinking dial is sourced from the environment so an operator can
    # retune without a code change. "medium" is a balanced starting
    # point; real engagements refine this in the first few weeks
    # against observed output quality vs. cost.
    #   L3_THINKING_EFFORT           "low" | "medium" | "high"   (preferred)
    #   L3_THINKING_BUDGET_TOKENS    integer                      (legacy fallback)
    thinking_effort = os.environ.get("L3_THINKING_EFFORT", "medium").strip() or None
    _legacy_budget = os.environ.get("L3_THINKING_BUDGET_TOKENS", "").strip()
    extended_thinking_budget_tokens = int(_legacy_budget) if _legacy_budget.isdigit() else None

    system_prompt = textwrap.dedent(
        """\
        You are the L3 strategic synthesis agent. You receive a set of
        L2 cross-entity analyses and produce ONE JSON Synthesis object
        containing a portfolio-level briefing for a human executive.

        Return EXACTLY ONE JSON object matching this schema — no
        markdown, no prose, no code fences:

        {
          "analysis_ids":               ["<uuid>", ...],
          "headline":                   "<=120 char executive headline",
          "briefing":                   "<3-6 paragraphs, under 4000 chars>",
          "contradictions_detected":    ["<contradiction>", ...],
          "suggested_next_watch":       ["<thing to watch>", ...]
        }

        What makes a good briefing:
          - Write for a non-technical executive of a multi-entity
            business. No jargon, no hedging, no repetition. Short
            sentences, strong verbs.
          - State the 2-3 things that actually matter THIS cycle across
            the portfolio, with the evidence in one line each.
          - Be explicit about contradictions: where does this cycle's
            data contradict a prior assumption, last cycle's analysis,
            or the stated operating strategy? List each as its own
            entry in contradictions_detected.
          - suggested_next_watch names the 1-5 concrete things to
            monitor in the next cycle. These should be observable
            signals, not vague themes ("checkout queue length 6pm-10pm
            on event weekends" not "checkout performance").
          - If the analyses you receive do not warrant executive
            attention, say so plainly in the briefing. Do not invent
            urgency.
          - Output ONLY the JSON. No preamble.
        """
    ).strip()

    def build_prompt(self, payload: L3Input) -> str:
        lines: list[str] = [
            f"Synthesize a portfolio briefing from these {len(payload.analyses)} "
            "L2 analyses (newest first):",
            "",
        ]
        # Cap at 20 even though L3Input does not — batches larger than
        # this consume too much Opus budget per call and produce worse
        # synthesis (dilution).
        for a in payload.analyses[:20]:
            lines.append(
                f"- analysis_id={a.id} "
                f"entities={[e.value for e in a.involves_entities]} "
                f"escalate_to_l3={a.escalate_to_l3} "
                f"pattern_summary={a.pattern_summary!r} "
                f"recommended_action={a.recommended_action!r}"
            )
        lines.extend(["", "Return the JSON Synthesis object now."])
        return "\n".join(lines)
