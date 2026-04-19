"""End-to-end demo run.

Spins up the full L1 → L2 → L3 → L4 pipeline against a handful of
hand-crafted events drawn from a generic multi-entity portfolio
(retail + hospitality + real_estate). Uses the real Anthropic API.

Prerequisites:
    - ANTHROPIC_API_KEY set in the environment (or in a local .env).
    - Optional: CONTEXT_STORE, SQLITE_PATH, HAIKU_MODEL, SONNET_MODEL,
      OPUS_MODEL, L3_THINKING_BUDGET_TOKENS.

Run:
    python examples/demo_run.py

What you will see:
    - Four LLM calls in the happy path (two L1 + one L2 + one L3).
    - A morning-brief-style synthesis printed to stdout via L4.
    - A line summarizing tokens, latency, and estimated cost.
    - A SQLite file at ./context/context.db containing every
      classification, analysis, synthesis, and audit row.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

# Make sure the repo root is importable when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.l1_classifier import L1Classifier  # noqa: E402
from agents.l2_analyzer import L2Analyzer  # noqa: E402
from agents.l3_synthesizer import L3Synthesizer  # noqa: E402
from agents.l4_delivery import Channel, L4Delivery  # noqa: E402
from context.store import SQLiteContextStore  # noqa: E402
from orchestrator.router import TieredOrchestrator  # noqa: E402
from schemas.models import EntityType, Event  # noqa: E402


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _configure_logging() -> None:
    level = _env("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        level=getattr(logging, level, logging.INFO),
    )


def _demo_events() -> list[Event]:
    """Six events spanning three entities, in the same 24-hour window."""
    base = datetime(2026, 4, 19, 8, 0, tzinfo=UTC)

    return [
        # Retail: a strong day at one retail location
        Event(
            source_system="retail_pos",
            entity_hint=EntityType.RETAIL,
            payload={
                "location_anon": "retail_A",
                "ticket_count": 58,
                "gross_revenue_usd": 34_200.00,
                "basket_size_avg_usd": 589.65,
            },
            observed_at=base,
        ),
        # Retail: a routine day at a second location
        Event(
            source_system="retail_pos",
            entity_hint=EntityType.RETAIL,
            payload={
                "location_anon": "retail_B",
                "ticket_count": 23,
                "gross_revenue_usd": 9_450.00,
                "basket_size_avg_usd": 410.87,
            },
            observed_at=base + timedelta(minutes=15),
        ),
        # Hospitality: a rapidly rising ops-ticket queue at a hotel
        Event(
            source_system="hotel_pms",
            entity_hint=EntityType.HOSPITALITY,
            payload={
                "location_anon": "property_A",
                "ops_tickets_open": 14,
                "ops_tickets_opened_last_24h": 11,
                "occupancy_pct": 91.4,
            },
            observed_at=base + timedelta(hours=2),
        ),
        # Hospitality: a batch of late check-outs at the same property
        Event(
            source_system="hotel_pms",
            entity_hint=EntityType.HOSPITALITY,
            payload={
                "location_anon": "property_A",
                "late_checkouts_count": 7,
                "auto_fee_applied_count": 2,
                "front_desk_overrides_count": 5,
            },
            observed_at=base + timedelta(hours=3),
        ),
        # External signal affecting the retail + hospitality zone
        Event(
            source_system="weather_service",
            entity_hint=EntityType.RETAIL,
            payload={
                "signal_type": "storm_advisory",
                "severity": "moderate",
                "eta_local": "14:00",
                "duration_hours": 8,
            },
            observed_at=base + timedelta(hours=4),
        ),
        # Real estate: an unusual maintenance surge on one property
        Event(
            source_system="building_maintenance",
            entity_hint=EntityType.REAL_ESTATE,
            payload={
                "location_anon": "building_C",
                "work_orders_open": 9,
                "hvac_related_count": 6,
                "after_hours_calls_count": 3,
            },
            observed_at=base + timedelta(hours=5),
        ),
    ]


def main() -> int:
    load_dotenv()
    _configure_logging()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env, "
            "fill in your key, and rerun.",
            file=sys.stderr,
        )
        return 2

    client = Anthropic(api_key=api_key)

    haiku = _env("HAIKU_MODEL", "claude-haiku-4-5-20251001")
    sonnet = _env("SONNET_MODEL", "claude-sonnet-4-6")
    opus = _env("OPUS_MODEL", "claude-opus-4-7")

    store = SQLiteContextStore(_env("SQLITE_PATH", "./context/context.db"))

    orch = TieredOrchestrator(
        l1=L1Classifier(client=client, model_id=haiku),
        l2=L2Analyzer(client=client, model_id=sonnet),
        l3=L3Synthesizer(client=client, model_id=opus),
        l4=L4Delivery(),
        store=store,
    )

    events = _demo_events()
    print(
        f"\nRunning demo over {len(events)} events spanning "
        "retail, hospitality, and real estate."
    )
    print(f"Models: {haiku} / {sonnet} / {opus}")
    print()

    result = orch.run_cycle(
        events, correlation_window_hours=24, deliver_to=Channel.STDOUT
    )

    print("\n----- Run summary -----")
    print(f"  Classifications: {len(result.classifications)}")
    print(f"  Analyses:        {len(result.analyses)}")
    print(f"  Synthesis:       {'yes' if result.synthesis else 'no'}")
    print(
        f"  L1/L2/L3 failures: "
        f"{result.l1_failures}/{result.l2_failures}/{result.l3_failures}"
    )
    print(f"  {result.cost_summary()}")
    print("\nAudit trail is in the SQLite database at ./context/context.db.")
    print(
        "Inspect with:\n    sqlite3 context/context.db 'SELECT "
        "agent_name, tier, input_tokens, output_tokens, "
        "estimated_cost_usd FROM audit_log;'"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
