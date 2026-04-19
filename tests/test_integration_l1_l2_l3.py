"""End-to-end integration tests for the tiered orchestrator.

Exercises the full L1 → L2 → L3 → L4 routing path against a mocked
Anthropic client and a real SQLite context store. Verifies that:

  - Every raw event is persisted.
  - L1 classifications land in the DB and are routed correctly.
  - L2 runs when there are notable events, skips when there are none.
  - L3 runs when at least one L2 analysis escalates.
  - The audit log reflects every LLM call.
  - Cost and latency are accumulated.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from agents.l1_classifier import L1Classifier
from agents.l2_analyzer import L2Analyzer
from agents.l3_synthesizer import L3Synthesizer
from agents.l4_delivery import DeliveryAdapter, DeliveryRecord, L4Delivery
from context.store import SQLiteContextStore
from orchestrator.router import TieredOrchestrator
from schemas.models import EntityType, Event

# ---------------------------------------------------------------------------
# Test-only delivery adapter that captures instead of printing
# ---------------------------------------------------------------------------


class _CapturingAdapter(DeliveryAdapter):
    def __init__(self) -> None:
        self.records: list[DeliveryRecord] = []

    def send(self, record: DeliveryRecord) -> None:
        self.records.append(record)


# ---------------------------------------------------------------------------
# Mock response builder that returns different JSON per model tier
# ---------------------------------------------------------------------------


def _make_tiered_client(
    *,
    l1_priority: str = "notable",
    l2_escalate: bool = True,
    l3_headline: str = "Cross-entity weekend brief",
):
    """Build a MagicMock whose ``messages.create`` returns the right
    JSON shape depending on which model it was asked to run.

    Recognizes the model by looking at ``model`` kwarg. The real code
    passes three different model ids for the three tiers; the mock
    keys its canned output off the substring.
    """
    from unittest.mock import MagicMock

    class _Block:
        def __init__(self, t: str) -> None:
            self.type = "text"
            self.text = t

    class _Usage:
        def __init__(self, i: int, o: int) -> None:
            self.input_tokens = i
            self.output_tokens = o

    class _Resp:
        def __init__(self, text: str, i: int = 120, o: int = 80) -> None:
            self.content = [_Block(text)]
            self.usage = _Usage(i, o)

    def _side_effect(**kwargs):  # type: ignore[no-untyped-def]
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        prompt_text = messages[0]["content"] if messages else ""

        # L1: classification keyed to the event_id it was asked about.
        if "haiku" in model.lower():
            # Yank the first uuid from the prompt so the classification
            # links to the actual event.
            import re
            match = re.search(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                prompt_text,
            )
            event_id = match.group(0) if match else str(uuid4())
            # Alternate entities so the batch is genuinely cross-entity.
            entity = "hospitality" if "hotel_pms" in prompt_text else "retail"
            body = {
                "event_id": event_id,
                "entity": entity,
                "category": "operations" if entity == "hospitality" else "sale",
                "priority": l1_priority,
                "rationale": "Outside normal band for this entity.",
            }
            return _Resp(json.dumps(body), i=100, o=60)

        # L2: analysis referencing two source events.
        if "sonnet" in model.lower():
            import re
            ids = re.findall(
                r"event_id=([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
                prompt_text,
            )
            source_ids = ids[:2] if ids else [str(uuid4())]
            body = {
                "source_event_ids": source_ids,
                "involves_entities": ["hospitality", "retail"],
                "pattern_summary": (
                    "Hospitality and retail signals in the same window "
                    "co-move: ops signal at the hotel coincides with a "
                    "retail sales lift. Cross-entity pattern would be "
                    "invisible if either business was viewed alone."
                ),
                "recommended_action": "Staff retail for +15% walk-ins tomorrow.",
                "escalate_to_l3": l2_escalate,
            }
            return _Resp(json.dumps(body), i=400, o=220)

        # L3: synthesis referencing the L2 analysis id.
        if "opus" in model.lower():
            import re
            ids = re.findall(
                r"analysis_id=([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
                prompt_text,
            )
            body = {
                "analysis_ids": ids[:1] if ids else [str(uuid4())],
                "headline": l3_headline,
                "briefing": (
                    "Portfolio summary. Two entities moved together "
                    "this cycle. Retail should prepare for the carry-"
                    "over tomorrow. Watch cancellation rate in parallel."
                ),
                "contradictions_detected": [
                    "Last cycle's assumption that retail leads did not hold.",
                ],
                "suggested_next_watch": [
                    "Hotel ops ticket volume next 24h.",
                    "Retail foot traffic vs baseline.",
                ],
            }
            return _Resp(json.dumps(body), i=1_200, o=800)

        raise AssertionError(f"Unexpected model in mock: {model!r}")

    client = MagicMock()
    client.messages.create.side_effect = _side_effect
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_store(tmp_path: Path) -> SQLiteContextStore:
    return SQLiteContextStore(tmp_path / "ctx.db")


@pytest.fixture()
def events() -> list[Event]:
    """Two events from different entities in the same window."""
    return [
        Event(
            source_system="retail_pos",
            entity_hint=EntityType.RETAIL,
            payload={"location_anon": "store_1", "amount_usd": 1850.0},
            observed_at=datetime(2026, 4, 18, 14, 30, tzinfo=UTC),
        ),
        Event(
            source_system="hotel_pms",
            entity_hint=EntityType.HOSPITALITY,
            payload={"location_anon": "property_a", "ops_ticket_count": 7},
            observed_at=datetime(2026, 4, 18, 15, 45, tzinfo=UTC),
        ),
    ]


@pytest.fixture()
def orchestrator(tmp_store):  # type: ignore[no-untyped-def]
    client = _make_tiered_client()
    return (
        TieredOrchestrator(
            l1=L1Classifier(client=client, model_id="claude-haiku-4-5"),
            l2=L2Analyzer(client=client, model_id="claude-sonnet-4-6"),
            l3=L3Synthesizer(client=client, model_id="claude-opus-4-7"),
            l4=L4Delivery(adapters=[_CapturingAdapter()]),
            store=tmp_store,
        ),
        tmp_store,
        client,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_full_cycle_produces_all_four_artifacts(orchestrator, events):
    orch, _store, _client = orchestrator

    result = orch.run_cycle(events)

    # Every event classified
    assert len(result.classifications) == len(events)
    # At least one analysis produced
    assert len(result.analyses) == 1
    # Synthesis produced because L2 escalated
    assert result.synthesis is not None
    assert result.synthesis.headline == "Cross-entity weekend brief"
    # No failures at any tier
    assert result.l1_failures == result.l2_failures == result.l3_failures == 0
    # Cost accumulated
    assert result.total_cost_usd > 0
    assert result.total_input_tokens > 0


def test_full_cycle_persists_artifacts_and_audit_log(orchestrator, events):
    orch, store, _client = orchestrator

    orch.run_cycle(events)

    with sqlite3.connect(store._db_path) as conn:
        ev_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        cls_count = conn.execute("SELECT COUNT(*) FROM classifications").fetchone()[0]
        ana_count = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        syn_count = conn.execute("SELECT COUNT(*) FROM syntheses").fetchone()[0]
        audit_rows = conn.execute(
            "SELECT tier, agent_name FROM audit_log ORDER BY id"
        ).fetchall()

    assert ev_count == 2
    assert cls_count == 2
    assert ana_count == 1
    assert syn_count == 1

    # One audit row per LLM call: 2 x L1 + 1 x L2 + 1 x L3 = 4 rows
    assert len(audit_rows) == 4
    tiers = [row[0] for row in audit_rows]
    assert tiers.count("haiku") == 2
    assert tiers.count("sonnet") == 1
    assert tiers.count("opus") == 1


# ---------------------------------------------------------------------------
# Routing contracts
# ---------------------------------------------------------------------------


def test_all_routine_events_skip_l2_and_l3(tmp_store, events):
    client = _make_tiered_client(l1_priority="routine")
    orch = TieredOrchestrator(
        l1=L1Classifier(client=client, model_id="claude-haiku-4-5"),
        l2=L2Analyzer(client=client, model_id="claude-sonnet-4-6"),
        l3=L3Synthesizer(client=client, model_id="claude-opus-4-7"),
        l4=L4Delivery(adapters=[_CapturingAdapter()]),
        store=tmp_store,
    )

    result = orch.run_cycle(events)

    assert len(result.classifications) == 2
    # No notable events → no L2 analysis → no L3 synthesis
    assert result.analyses == []
    assert result.synthesis is None


def test_l2_no_escalation_skips_l3(tmp_store, events):
    client = _make_tiered_client(l2_escalate=False)
    orch = TieredOrchestrator(
        l1=L1Classifier(client=client, model_id="claude-haiku-4-5"),
        l2=L2Analyzer(client=client, model_id="claude-sonnet-4-6"),
        l3=L3Synthesizer(client=client, model_id="claude-opus-4-7"),
        l4=L4Delivery(adapters=[_CapturingAdapter()]),
        store=tmp_store,
    )

    result = orch.run_cycle(events)

    assert len(result.analyses) == 1
    assert result.analyses[0].escalate_to_l3 is False
    assert result.synthesis is None


def test_delivery_adapter_receives_synthesis(tmp_store, events):
    captured = _CapturingAdapter()
    client = _make_tiered_client(l3_headline="Live delivery test")
    orch = TieredOrchestrator(
        l1=L1Classifier(client=client, model_id="claude-haiku-4-5"),
        l2=L2Analyzer(client=client, model_id="claude-sonnet-4-6"),
        l3=L3Synthesizer(client=client, model_id="claude-opus-4-7"),
        l4=L4Delivery(adapters=[captured]),
        store=tmp_store,
    )

    orch.run_cycle(events)

    assert len(captured.records) == 1
    assert captured.records[0].subject == "Live delivery test"
