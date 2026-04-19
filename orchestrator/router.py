"""Tiered orchestrator.

Routes a batch of raw events through the L1 → L2 → L3 → L4 pipeline,
writing every decision to the shared context store so every output is
auditable end-to-end.

This is the concrete demonstration of the "no black box" promise: any
output the system produces can be traced by ID back to the L3
synthesis, which is tied by analysis_ids to L2, which is tied by
source_event_ids to L1, which is tied by event_id to the raw event.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from agents.base import Agent, AgentResult
from agents.l1_classifier import L1Classifier
from agents.l2_analyzer import L2Analyzer, L2Input
from agents.l3_synthesizer import L3Input, L3Synthesizer
from agents.l4_delivery import Channel, L4Delivery
from context.store import ContextStore
from schemas.models import (
    Analysis,
    Classification,
    Event,
    EventPriority,
    Synthesis,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OrchestratorResult:
    """Summary of one end-to-end pipeline run.

    The caller gets back the artifacts produced at each tier plus a
    rolling cost summary. Every artifact has also been persisted to
    the context store by the orchestrator.
    """

    classifications: list[Classification] = field(default_factory=list)
    analyses: list[Analysis] = field(default_factory=list)
    synthesis: Synthesis | None = None

    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: int = 0

    l1_failures: int = 0
    l2_failures: int = 0
    l3_failures: int = 0

    def cost_summary(self) -> str:
        return (
            f"tokens in/out: {self.total_input_tokens}/{self.total_output_tokens} · "
            f"latency: {self.total_latency_ms} ms · "
            f"est. cost: ${self.total_cost_usd:.5f}"
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class TieredOrchestrator:
    """End-to-end router across the four agents.

    Not an LLM-backed agent itself — it is a plain coordinator. The
    orchestrator knows nothing about prompts, only about which agent
    consumes which tier's output. Adding an agent means registering it
    into the pipeline; it does not mean changing the orchestrator's
    routing logic for the existing agents.
    """

    def __init__(
        self,
        *,
        l1: L1Classifier,
        l2: L2Analyzer,
        l3: L3Synthesizer,
        l4: L4Delivery,
        store: ContextStore,
    ) -> None:
        self._l1 = l1
        self._l2 = l2
        self._l3 = l3
        self._l4 = l4
        self._store = store

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        events: Sequence[Event],
        *,
        correlation_window_hours: int = 24,
        deliver_to: Channel = Channel.STDOUT,
    ) -> OrchestratorResult:
        """Run one end-to-end cycle over a batch of events."""
        result = OrchestratorResult()

        # --- Stage 1: persist raw events -----------------------------
        # save_event is idempotent — reprocessing a cycle with the same
        # event ids is a no-op rather than a crash. An IntegrityError
        # from a duplicate primary key is logged and swallowed; the rest
        # of the event list is still processed.
        for event in events:
            try:
                self._store.save_event(event)
            except Exception:
                logger.debug("save_event skipped for %s (likely duplicate)", event.id)

        # --- Stage 2: L1 classify every event ------------------------
        notable_classifications: list[Classification] = []
        urgent_classifications: list[Classification] = []

        for event in events:
            try:
                agent_result = self._l1.run(event)
            except ValidationError:
                result.l1_failures += 1
                continue
            except Exception:
                logger.exception("L1 failed on event %s", event.id)
                result.l1_failures += 1
                continue

            self._track(result, agent_result, agent=self._l1)

            classification = agent_result.output
            self._store.save_classification(
                classification,
                model_id=agent_result.model_id,
                tier=self._l1.tier.value,
                agent_name=type(self._l1).__name__,
                input_tokens=agent_result.input_tokens,
                output_tokens=agent_result.output_tokens,
                estimated_cost_usd=agent_result.estimated_cost_usd,
                latency_ms=agent_result.latency_ms,
            )
            result.classifications.append(classification)

            if classification.priority is EventPriority.NOTABLE:
                notable_classifications.append(classification)
            elif classification.priority is EventPriority.URGENT:
                # Urgent bypasses L2 — these get routed straight to L3
                # via a synthetic one-event analysis. We also keep them
                # in the notable bucket so L2 still sees them in the
                # batch for context.
                urgent_classifications.append(classification)
                notable_classifications.append(classification)

        # --- Stage 3: L2 cross-entity analysis -----------------------
        if notable_classifications:
            try:
                l2_result = self._l2.run(
                    L2Input(
                        classifications=notable_classifications,
                        correlation_window_hours=correlation_window_hours,
                    )
                )
                self._track(result, l2_result, agent=self._l2)
                analysis = l2_result.output
                self._store.save_analysis(
                    analysis,
                    model_id=l2_result.model_id,
                    tier=self._l2.tier.value,
                    agent_name=type(self._l2).__name__,
                    input_tokens=l2_result.input_tokens,
                    output_tokens=l2_result.output_tokens,
                    estimated_cost_usd=l2_result.estimated_cost_usd,
                    latency_ms=l2_result.latency_ms,
                )
                result.analyses.append(analysis)
            except ValidationError:
                result.l2_failures += 1
            except Exception:
                logger.exception("L2 failed on notable batch")
                result.l2_failures += 1

        # --- Stage 4: L3 strategic synthesis -------------------------
        # Gate: L3 runs when either
        #   (a) at least one L2 analysis flagged itself for escalation,
        #       OR
        #   (b) any event was classified as URGENT and L2 produced at
        #       least one analysis this cycle (we reuse L2's synthesis
        #       input rather than synthesize a single-event Analysis —
        #       L2 already saw the urgent event in its notable batch).
        escalated = [a for a in result.analyses if a.escalate_to_l3]
        if not escalated and urgent_classifications and result.analyses:
            # Urgent override: even if L2 didn't recommend escalation,
            # route to L3 so a human operator gets notified today.
            escalated = list(result.analyses)
        if escalated:
            try:
                l3_result = self._l3.run(L3Input(analyses=escalated))
                self._track(result, l3_result, agent=self._l3)
                synthesis = l3_result.output
                self._store.save_synthesis(
                    synthesis,
                    model_id=l3_result.model_id,
                    tier=self._l3.tier.value,
                    agent_name=type(self._l3).__name__,
                    input_tokens=l3_result.input_tokens,
                    output_tokens=l3_result.output_tokens,
                    estimated_cost_usd=l3_result.estimated_cost_usd,
                    latency_ms=l3_result.latency_ms,
                )
                result.synthesis = synthesis
            except ValidationError:
                result.l3_failures += 1
            except Exception:
                logger.exception("L3 failed on escalated analyses")
                result.l3_failures += 1

        # --- Stage 5: L4 delivery ------------------------------------
        if result.synthesis is not None:
            self._l4.dispatch_synthesis(result.synthesis, channel=deliver_to)

        logger.info("cycle complete · %s", result.cost_summary())
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _track(
        result: OrchestratorResult,
        agent_result: AgentResult[Any],
        *,
        agent: Agent[Any, Any],
    ) -> None:
        result.total_cost_usd += agent_result.estimated_cost_usd
        result.total_input_tokens += agent_result.input_tokens
        result.total_output_tokens += agent_result.output_tokens
        result.total_latency_ms += agent_result.latency_ms
        logger.debug(
            "%s run · in=%d out=%d cost=%.5f latency=%d ms",
            type(agent).__name__,
            agent_result.input_tokens,
            agent_result.output_tokens,
            agent_result.estimated_cost_usd,
            agent_result.latency_ms,
        )
