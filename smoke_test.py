"""Self-contained smoke test runner.

Exercises the same invariants as the pytest suite but without pytest
— useful for environments where pytest discovery misbehaves.

Run: python smoke_test.py
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

from pydantic import ValidationError

from agents.l1_classifier import L1Classifier
from schemas.models import (
    Analysis,
    Classification,
    EntityType,
    Event,
    EventCategory,
    EventPriority,
    Synthesis,
)

_PASS = 0
_FAIL = 0


def _check(name: str, fn) -> None:  # type: ignore[no-untyped-def]
    global _PASS, _FAIL
    try:
        fn()
        print(f"  [PASS] {name}")
        _PASS += 1
    except AssertionError as e:
        print(f"  [FAIL] {name}: {e}")
        _FAIL += 1
    except Exception:
        print(f"  [ERROR] {name}:")
        traceback.print_exc()
        _FAIL += 1


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_event_rejects_extra_fields() -> None:
    try:
        Event(
            source_system="pos",
            payload={"a": 1},
            observed_at=datetime.now(UTC),
            made_up_field="nope",  # type: ignore[call-arg]
        )
    except ValidationError:
        return
    raise AssertionError("Event accepted an unknown field")


def test_classification_requires_nonempty_rationale() -> None:
    try:
        Classification(
            event_id=uuid4(),
            entity=EntityType.RETAIL,
            category=EventCategory.SALE,
            priority=EventPriority.ROUTINE,
            rationale="",
        )
    except ValidationError:
        return
    raise AssertionError("Classification accepted empty rationale")


def test_analysis_requires_event_and_entity() -> None:
    try:
        Analysis(
            source_event_ids=[],
            involves_entities=[],
            pattern_summary="x",
            escalate_to_l3=False,
        )
    except ValidationError:
        return
    raise AssertionError("Analysis accepted empty lists")


def test_synthesis_rejects_empty_briefing() -> None:
    try:
        Synthesis(analysis_ids=[uuid4()], headline="h", briefing="")
    except ValidationError:
        return
    raise AssertionError("Synthesis accepted empty briefing")


def test_classification_roundtrips_through_json() -> None:
    original = Classification(
        event_id=uuid4(),
        entity=EntityType.HOSPITALITY,
        category=EventCategory.OPERATIONS,
        priority=EventPriority.NOTABLE,
        rationale="Roundtrip test.",
    )
    restored = Classification.model_validate_json(original.model_dump_json())
    assert restored == original


# ---------------------------------------------------------------------------
# L1 classifier — with mocked Anthropic client
# ---------------------------------------------------------------------------


def _fake_response(text: str, *, input_tokens: int = 100, output_tokens: int = 50):
    class _Block:
        def __init__(self, t: str) -> None:
            self.type = "text"
            self.text = t

    class _Usage:
        def __init__(self, i: int, o: int) -> None:
            self.input_tokens = i
            self.output_tokens = o

    class _Resp:
        def __init__(self) -> None:
            self.content = [_Block(text)]
            self.usage = _Usage(input_tokens, output_tokens)

    return _Resp()


def _event() -> Event:
    return Event(
        source_system="jewelry_pos",
        entity_hint=EntityType.RETAIL,
        payload={"amount_usd": 412.50, "location_anon": "store_2"},
        observed_at=datetime(2026, 4, 17, 14, 3, tzinfo=UTC),
    )


def test_classifier_happy_path() -> None:
    client = MagicMock()
    event = _event()
    client.messages.create.return_value = _fake_response(
        json.dumps(
            {
                "event_id": str(event.id),
                "entity": "retail",
                "category": "sale",
                "priority": "routine",
                "rationale": "Typical POS transaction.",
            }
        )
    )
    c = L1Classifier(client=client, model_id="claude-haiku-4-5")
    result = c.run(event)
    assert result.output.entity is EntityType.RETAIL
    assert result.output.priority is EventPriority.ROUTINE
    assert result.input_tokens == 100
    assert result.output_tokens == 50


def test_classifier_accepts_fenced_json() -> None:
    client = MagicMock()
    event = _event()
    body = {
        "event_id": str(event.id),
        "entity": "retail",
        "category": "sale",
        "priority": "notable",
        "rationale": "Above-typical amount.",
    }
    client.messages.create.return_value = _fake_response(
        f"```json\n{json.dumps(body)}\n```"
    )
    c = L1Classifier(client=client, model_id="claude-haiku-4-5")
    result = c.run(event)
    assert result.output.priority is EventPriority.NOTABLE


def test_classifier_rejects_malformed_output() -> None:
    client = MagicMock()
    event = _event()
    client.messages.create.return_value = _fake_response(
        json.dumps(
            {
                "event_id": str(event.id),
                "entity": "retail",
                "category": "sale",
                # missing priority
                "rationale": "oops",
            }
        )
    )
    c = L1Classifier(client=client, model_id="claude-haiku-4-5")
    try:
        c.run(event)
    except ValidationError:
        return
    raise AssertionError("classifier accepted malformed LLM output")


def test_classifier_cost_math() -> None:
    client = MagicMock()
    event = _event()
    client.messages.create.return_value = _fake_response(
        json.dumps(
            {
                "event_id": str(event.id),
                "entity": "retail",
                "category": "sale",
                "priority": "routine",
                "rationale": "ok",
            }
        ),
        input_tokens=1_000,
        output_tokens=500,
    )
    c = L1Classifier(client=client, model_id="claude-haiku-4-5")
    result = c.run(event)
    expected = (1_000 * 0.25 + 500 * 1.25) / 1_000_000
    assert abs(result.estimated_cost_usd - expected) < 1e-9


def test_build_prompt_includes_metadata() -> None:
    c = L1Classifier(client=None, model_id="claude-haiku-4-5")  # type: ignore[arg-type]
    event = _event()
    prompt = c.build_prompt(event)
    assert str(event.id) in prompt
    assert "jewelry_pos" in prompt
    assert "Return the JSON object now." in prompt


# ---------------------------------------------------------------------------
# SQLite context store
# ---------------------------------------------------------------------------


def test_sqlite_store_roundtrip(tmp_path="") -> None:  # type: ignore[no-untyped-def]
    import os as _os
    import tempfile

    from context.store import SQLiteContextStore

    with tempfile.TemporaryDirectory() as td:
        path = _os.path.join(td, "smoke.db")
        store = SQLiteContextStore(path)

        event = _event()
        store.save_event(event)

        classification = Classification(
            event_id=event.id,
            entity=EntityType.RETAIL,
            category=EventCategory.SALE,
            priority=EventPriority.NOTABLE,
            rationale="Smoke-test write path.",
        )
        store.save_classification(
            classification,
            model_id="claude-haiku-4-5",
            tier="haiku",
            agent_name="L1Classifier",
            input_tokens=100,
            output_tokens=50,
            estimated_cost_usd=0.00009,
            latency_ms=420,
        )

        recent = store.recent_notable_classifications(limit=10)
        assert len(recent) == 1
        assert recent[0].event_id == event.id


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> int:
    print("Schema validation")
    _check("Event rejects extra fields",              test_event_rejects_extra_fields)
    _check(
        "Classification requires rationale",
        test_classification_requires_nonempty_rationale,
    )
    _check("Analysis requires event + entity",        test_analysis_requires_event_and_entity)
    _check("Synthesis rejects empty briefing",        test_synthesis_rejects_empty_briefing)
    _check("Classification roundtrips JSON",          test_classification_roundtrips_through_json)

    print("\nL1 classifier")
    _check("Happy path returns typed output",         test_classifier_happy_path)
    _check("Accepts fenced JSON",                     test_classifier_accepts_fenced_json)
    _check("Rejects malformed output",                test_classifier_rejects_malformed_output)
    _check("Cost math matches Haiku rate card",       test_classifier_cost_math)
    _check("Prompt includes event metadata",          test_build_prompt_includes_metadata)

    print("\nContext store")
    _check("SQLite roundtrip with audit log",         test_sqlite_store_roundtrip)

    print(f"\n{_PASS} passed, {_FAIL} failed.")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
