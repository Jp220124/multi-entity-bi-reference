"""Regression tests for bugs caught in the senior-grade review.

Each test corresponds to a specific defect. If any of these tests
regress, the named defect has returned.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

from agents.l1_classifier import L1Classifier
from agents.l2_analyzer import L2Analyzer
from agents.l3_synthesizer import L3Synthesizer
from agents.l4_delivery import L4Delivery
from context.store import SQLiteContextStore
from orchestrator.router import TieredOrchestrator
from schemas.models import EntityType, Event

# ---------------------------------------------------------------------------
# Helpers shared with the integration tests' mock builder pattern
# ---------------------------------------------------------------------------


class _Block:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _Resp:
    def __init__(self, text: str, i: int = 100, o: int = 50) -> None:
        self.content = [_Block(text)]
        self.usage = _Usage(i, o)


def _make_client(l1_priority: str, l2_escalate: bool | None = True):
    """Build a mocked Anthropic client that returns tier-specific JSON."""
    import re

    def _side_effect(**kwargs):  # type: ignore[no-untyped-def]
        model = kwargs.get("model", "").lower()
        prompt = kwargs.get("messages", [{}])[0].get("content", "")

        if "haiku" in model:
            match = re.search(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                prompt,
            )
            event_id = match.group(0) if match else "00000000-0000-0000-0000-000000000000"
            return _Resp(
                json.dumps(
                    {
                        "event_id": event_id,
                        "entity": "retail",
                        "category": "sale",
                        "priority": l1_priority,
                        "rationale": "Test fixture response.",
                    }
                )
            )

        if "sonnet" in model:
            ids = re.findall(
                r"event_id=([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
                prompt,
            )
            return _Resp(
                json.dumps(
                    {
                        "source_event_ids": ids[:1] or ["00000000-0000-0000-0000-000000000000"],
                        "involves_entities": ["retail", "hospitality"],
                        "pattern_summary": "Test cross-entity pattern.",
                        "recommended_action": "Test action.",
                        "escalate_to_l3": l2_escalate if l2_escalate is not None else False,
                    }
                )
            )

        if "opus" in model:
            ids = re.findall(
                r"analysis_id=([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
                prompt,
            )
            return _Resp(
                json.dumps(
                    {
                        "analysis_ids": ids[:1] or ["00000000-0000-0000-0000-000000000000"],
                        "headline": "Test synthesis",
                        "briefing": "Test synthesis briefing produced by the regression fixture.",
                        "contradictions_detected": [],
                        "suggested_next_watch": ["item_a"],
                    }
                )
            )

        raise AssertionError(f"Unexpected model: {model!r}")

    client = MagicMock()
    client.messages.create.side_effect = _side_effect
    return client


def _build_orchestrator(client, store):  # type: ignore[no-untyped-def]
    return TieredOrchestrator(
        l1=L1Classifier(client=client, model_id="claude-haiku-4-5"),
        l2=L2Analyzer(client=client, model_id="claude-sonnet-4-6"),
        l3=L3Synthesizer(client=client, model_id="claude-opus-4-7"),
        l4=L4Delivery(),
        store=store,
    )


# ---------------------------------------------------------------------------
# F-1: save_event is idempotent (replay safety)
# ---------------------------------------------------------------------------


def test_save_event_idempotent_does_not_crash_on_duplicate_id(tmp_path: Path) -> None:
    store = SQLiteContextStore(tmp_path / "ctx.db")
    event = Event(
        source_system="retail_pos",
        entity_hint=EntityType.RETAIL,
        payload={"amount_usd": 42.0},
        observed_at=datetime(2026, 4, 19, 10, 0, tzinfo=UTC),
    )

    # First insert should succeed.
    store.save_event(event)
    # Reinsert of the same event must NOT raise.
    store.save_event(event)
    store.save_event(event)


def test_run_cycle_tolerates_reprocessing_same_events(tmp_path: Path) -> None:
    """If the orchestrator is re-run on the same event set, it should
    not crash on the duplicate-event-insert. Prior behaviour raised
    sqlite3.IntegrityError and aborted the whole cycle.
    """
    store = SQLiteContextStore(tmp_path / "ctx.db")
    client = _make_client(l1_priority="notable")
    orch = _build_orchestrator(client, store)

    event = Event(
        source_system="retail_pos",
        entity_hint=EntityType.RETAIL,
        payload={"amount_usd": 42.0},
        observed_at=datetime(2026, 4, 19, 10, 0, tzinfo=UTC),
    )

    r1 = orch.run_cycle([event])
    r2 = orch.run_cycle([event])  # replay

    # Both cycles complete without crashing.
    assert len(r1.classifications) == 1
    assert len(r2.classifications) == 1


# ---------------------------------------------------------------------------
# A-1: L3 urgent-override now reaches L3 reliably when L2 produced any analysis
# ---------------------------------------------------------------------------


def test_l3_runs_when_urgent_event_and_l2_did_not_self_escalate(tmp_path: Path) -> None:
    """Previously: if L2 produced a non-escalating analysis AND an
    event was urgent, the L3 synthesis was silently skipped because
    the bypass branch reassigned an empty list. Now: L3 must run.
    """
    store = SQLiteContextStore(tmp_path / "ctx.db")
    # urgent L1 priority forces urgent path; L2 says escalate_to_l3=False.
    client = _make_client(l1_priority="urgent", l2_escalate=False)
    orch = _build_orchestrator(client, store)

    event = Event(
        source_system="retail_pos",
        entity_hint=EntityType.RETAIL,
        payload={"amount_usd": 9_999.0},
        observed_at=datetime(2026, 4, 19, 10, 0, tzinfo=UTC),
    )

    result = orch.run_cycle([event])

    # An L2 analysis was produced (escalate_to_l3=False per fixture)...
    assert len(result.analyses) == 1
    assert result.analyses[0].escalate_to_l3 is False
    # ...and L3 still ran because the event was urgent.
    assert result.synthesis is not None, "L3 must run when urgent events are present"


# ---------------------------------------------------------------------------
# F-4: L3 thinking dial is resolved at construction, not at import time
# ---------------------------------------------------------------------------


def test_l3_thinking_effort_reflects_env_at_construction_time() -> None:
    """If load_dotenv() runs after the module is imported, the L3 class
    previously locked in the default "medium" at class-load time. Now
    the env is read on every ``L3Synthesizer(...)`` construction.
    """
    original = os.environ.get("L3_THINKING_EFFORT")
    try:
        os.environ["L3_THINKING_EFFORT"] = "high"
        high = L3Synthesizer(client=MagicMock(), model_id="claude-opus-4-7")
        assert high.thinking_effort == "high"

        os.environ["L3_THINKING_EFFORT"] = "low"
        low = L3Synthesizer(client=MagicMock(), model_id="claude-opus-4-7")
        assert low.thinking_effort == "low"
    finally:
        if original is None:
            os.environ.pop("L3_THINKING_EFFORT", None)
        else:
            os.environ["L3_THINKING_EFFORT"] = original


def test_l3_legacy_thinking_budget_parses_integer_only() -> None:
    """Non-integer values for the legacy budget env must not crash."""
    original = os.environ.get("L3_THINKING_BUDGET_TOKENS")
    try:
        os.environ["L3_THINKING_BUDGET_TOKENS"] = "not_a_number"
        synth = L3Synthesizer(client=MagicMock(), model_id="claude-opus-4-7")
        assert synth.extended_thinking_budget_tokens is None

        os.environ["L3_THINKING_BUDGET_TOKENS"] = "6000"
        synth2 = L3Synthesizer(client=MagicMock(), model_id="claude-opus-4-7")
        assert synth2.extended_thinking_budget_tokens == 6000
    finally:
        if original is None:
            os.environ.pop("L3_THINKING_BUDGET_TOKENS", None)
        else:
            os.environ["L3_THINKING_BUDGET_TOKENS"] = original
