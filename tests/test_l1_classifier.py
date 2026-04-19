"""Unit tests for the L1 classifier.

We do not hit the real Anthropic API here — the classifier is tested
against a mocked client that returns canned JSON. Schema validation
is exercised via the ``test_schema_validation.py`` module.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agents.l1_classifier import L1Classifier
from schemas.models import EntityType, EventCategory, EventPriority

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_classifier_returns_validated_classification(retail_sale_event, anthropic_mock):
    client, set_response = anthropic_mock
    set_response(
        json.dumps(
            {
                "event_id": str(retail_sale_event.id),
                "entity": "retail",
                "category": "sale",
                "priority": "routine",
                "rationale": "Typical jewelry retail POS transaction within expected range.",
            }
        )
    )

    classifier = L1Classifier(client=client, model_id="claude-haiku-4-5")
    result = classifier.run(retail_sale_event)

    assert result.output.entity is EntityType.RETAIL
    assert result.output.category is EventCategory.SALE
    assert result.output.priority is EventPriority.ROUTINE
    assert result.output.event_id == retail_sale_event.id


def test_classifier_accepts_fenced_json_response(hotel_cancellation_event, anthropic_mock):
    """LLMs sometimes wrap JSON in ``` fences; we should tolerate that."""
    client, set_response = anthropic_mock
    payload = {
        "event_id": str(hotel_cancellation_event.id),
        "entity": "hospitality",
        "category": "customer",
        "priority": "urgent",
        "rationale": "42 room-nights cancelled within 18 hours; revenue exposure.",
    }
    set_response(f"```json\n{json.dumps(payload)}\n```")

    classifier = L1Classifier(client=client, model_id="claude-haiku-4-5")
    result = classifier.run(hotel_cancellation_event)

    assert result.output.priority is EventPriority.URGENT
    assert result.output.entity is EntityType.HOSPITALITY


# ---------------------------------------------------------------------------
# Cost / metrics path
# ---------------------------------------------------------------------------


def test_classifier_reports_cost_and_latency(retail_sale_event, anthropic_mock):
    client, set_response = anthropic_mock
    set_response(
        json.dumps(
            {
                "event_id": str(retail_sale_event.id),
                "entity": "retail",
                "category": "sale",
                "priority": "routine",
                "rationale": "Routine transaction.",
            }
        ),
        input_tokens=1_000,
        output_tokens=500,
    )

    classifier = L1Classifier(client=client, model_id="claude-haiku-4-5")
    result = classifier.run(retail_sale_event)

    # Rough sanity check: cost follows the Haiku tier rate card exactly.
    # 1000 input @ $0.25 / 1M + 500 output @ $1.25 / 1M
    expected = (1_000 * 0.25 + 500 * 1.25) / 1_000_000
    assert result.estimated_cost_usd == pytest.approx(expected, rel=1e-6)
    assert result.input_tokens == 1_000
    assert result.output_tokens == 500
    assert result.latency_ms >= 0
    assert result.model_id == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Invariant: bad LLM output never produces a Classification
# ---------------------------------------------------------------------------


def test_classifier_rejects_malformed_output(retail_sale_event, anthropic_mock):
    client, set_response = anthropic_mock
    # Missing required field: priority.
    set_response(
        json.dumps(
            {
                "event_id": str(retail_sale_event.id),
                "entity": "retail",
                "category": "sale",
                "rationale": "Missing priority field.",
            }
        )
    )

    classifier = L1Classifier(client=client, model_id="claude-haiku-4-5")
    with pytest.raises(ValidationError):
        classifier.run(retail_sale_event)


def test_classifier_rejects_enum_violation(retail_sale_event, anthropic_mock):
    client, set_response = anthropic_mock
    # Made-up entity value.
    set_response(
        json.dumps(
            {
                "event_id": str(retail_sale_event.id),
                "entity": "interstellar",
                "category": "sale",
                "priority": "routine",
                "rationale": "Invalid entity.",
            }
        )
    )

    classifier = L1Classifier(client=client, model_id="claude-haiku-4-5")
    with pytest.raises(ValidationError):
        classifier.run(retail_sale_event)


# ---------------------------------------------------------------------------
# Prompt-building contract
# ---------------------------------------------------------------------------


def test_build_prompt_includes_event_metadata(retail_sale_event):
    # We can construct the agent without a live client for prompt-only tests.
    classifier = L1Classifier(client=None, model_id="claude-haiku-4-5")  # type: ignore[arg-type]
    prompt = classifier.build_prompt(retail_sale_event)

    assert str(retail_sale_event.id) in prompt
    assert "jewelry_pos" in prompt
    assert "retail" in prompt
    assert "Return the JSON object now." in prompt
