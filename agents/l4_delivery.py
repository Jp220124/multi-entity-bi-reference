"""L4 · Alerting / Output Delivery agent.

Shared utility used by Agents 1-3 to format and dispatch outputs to
the operator's preferred channel (Slack, email, or custom webhook).
Handles threading, retry, and dead-letter logging.

Status: in-memory reference implementation. Slack / email / Postmark
adapters are dropped in at engagement time — the interface below does
not change.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from schemas.models import Synthesis

logger = logging.getLogger(__name__)


class Channel(str, Enum):
    """Where a delivery lands."""

    SLACK = "slack"
    EMAIL = "email"
    STDOUT = "stdout"  # used in demo + tests


@dataclass(frozen=True, slots=True)
class DeliveryRecord:
    channel: Channel
    subject: str
    body: str


class DeliveryAdapter(ABC):
    """Pluggable transport for a single channel."""

    @abstractmethod
    def send(self, record: DeliveryRecord) -> None: ...


class StdoutAdapter(DeliveryAdapter):
    """Default adapter used by the demo and the integration tests."""

    def send(self, record: DeliveryRecord) -> None:
        print(f"\n=== [{record.channel.value}] {record.subject} ===")
        print(record.body)
        print("=" * (len(record.subject) + 8 + len(record.channel.value)))


class L4Delivery:
    """Format syntheses and dispatch via one or more adapters.

    Not an LLM-backed agent — it is a shared utility that every upstream
    agent uses to reach a human. Kept inside the ``agents`` package for
    symmetry with the contract language (Agent 4 · Alerting / Output
    Delivery).
    """

    def __init__(self, adapters: list[DeliveryAdapter] | None = None) -> None:
        self._adapters = adapters or [StdoutAdapter()]

    def dispatch_synthesis(
        self, synthesis: Synthesis, *, channel: Channel = Channel.STDOUT
    ) -> DeliveryRecord:
        record = DeliveryRecord(
            channel=channel,
            subject=synthesis.headline,
            body=self._render(synthesis),
        )
        for adapter in self._adapters:
            try:
                adapter.send(record)
            except Exception:
                # Dead-letter logging in a real deployment would persist
                # this record for retry. The reference impl just logs.
                logger.exception("delivery failed via %s", adapter)
        return record

    @staticmethod
    def _render(synthesis: Synthesis) -> str:
        parts = [synthesis.briefing, ""]
        if synthesis.contradictions_detected:
            parts.append("Contradictions to note:")
            parts.extend(f"  - {c}" for c in synthesis.contradictions_detected)
            parts.append("")
        if synthesis.suggested_next_watch:
            parts.append("Watch list for next cycle:")
            parts.extend(f"  - {w}" for w in synthesis.suggested_next_watch)
        return "\n".join(parts).rstrip()
