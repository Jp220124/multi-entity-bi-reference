"""Shared pytest fixtures.

Keeps tests offline by default — every fixture either constructs
in-memory objects or mocks the Anthropic client. Tests that exercise
real API calls live under ``tests/integration/`` and are skipped unless
``ANTHROPIC_API_KEY`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from schemas.models import EntityType, Event

# ---------------------------------------------------------------------------
# Event fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def retail_sale_event() -> Event:
    """A routine retail POS sale."""
    # retail_pos is a generic source name; a real deployment maps this
    # to the client's actual POS product (Square, Lightspeed, Shopify, etc.).
    return Event(
        source_system="retail_pos",
        entity_hint=EntityType.RETAIL,
        payload={
            "ticket_id_anon": "T-8921",
            "location_anon": "store_2",
            "amount_usd": 412.50,
            "sku_count": 2,
        },
        observed_at=datetime(2026, 4, 17, 14, 3, tzinfo=UTC),
    )


@pytest.fixture()
def hotel_cancellation_event() -> Event:
    """An urgent hospitality signal."""
    return Event(
        source_system="hotel_pms",
        entity_hint=EntityType.HOSPITALITY,
        payload={
            "reservation_id_anon": "R-55110",
            "location_anon": "property_a",
            "room_nights_lost": 42,
            "cancellation_window_hours": 18,
        },
        observed_at=datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# Anthropic client mock helpers
# ---------------------------------------------------------------------------


@dataclass
class _StubBlock:
    type: str
    text: str


@dataclass
class _StubUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class _StubResponse:
    content: list[_StubBlock]
    usage: _StubUsage


def _make_stub_response(
    *,
    text: str,
    input_tokens: int = 120,
    output_tokens: int = 80,
) -> _StubResponse:
    return _StubResponse(
        content=[_StubBlock(type="text", text=text)],
        usage=_StubUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


@pytest.fixture()
def anthropic_mock() -> tuple[MagicMock, Any]:
    """Build a mock Anthropic client whose ``messages.create`` is
    replaced per-test by calling ``set_response(text)``.
    """
    client = MagicMock()

    def set_response(text: str, *, input_tokens: int = 120, output_tokens: int = 80) -> None:
        client.messages.create.return_value = _make_stub_response(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    return client, set_response
