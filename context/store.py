"""Shared context store interface and SQLite implementation.

The interface deliberately stays narrow — a handful of typed read and
write methods. A production deployment swaps the SQLite implementation
for Supabase or Postgres by subclassing ``ContextStore`` and pointing
``CONTEXT_STORE=supabase`` in the environment.

Nothing in the agent code depends on SQLite specifically.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID

from schemas.models import (
    Analysis,
    Classification,
    EntityType,
    Event,
    EventPriority,
    Synthesis,
)

logger = logging.getLogger(__name__)


class ContextStore(ABC):
    """Storage interface for every tier.

    Implementations must be thread-safe for reads and serialise writes
    internally. The orchestrator runs agents sequentially in this
    reference implementation, so no write contention is expected —
    keeping the contract explicit lets a future concurrent orchestrator
    swap in without code changes above this layer.
    """

    # -- writes --------------------------------------------------------

    @abstractmethod
    def save_event(self, event: Event) -> None: ...

    @abstractmethod
    def save_classification(
        self,
        classification: Classification,
        *,
        model_id: str,
        tier: str,
        agent_name: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
        latency_ms: int,
    ) -> None: ...

    @abstractmethod
    def save_analysis(
        self,
        analysis: Analysis,
        *,
        model_id: str,
        tier: str,
        agent_name: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
        latency_ms: int,
    ) -> None: ...

    @abstractmethod
    def save_synthesis(
        self,
        synthesis: Synthesis,
        *,
        model_id: str,
        tier: str,
        agent_name: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
        latency_ms: int,
    ) -> None: ...

    # -- reads ---------------------------------------------------------

    @abstractmethod
    def recent_notable_classifications(
        self, *, limit: int = 50
    ) -> list[Classification]: ...

    @abstractmethod
    def recent_analyses(
        self,
        *,
        entities: list[EntityType] | None = None,
        limit: int = 20,
    ) -> list[Analysis]: ...


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------


class SQLiteContextStore(ContextStore):
    """SQLite-backed context store for local development and tests.

    The schema is in ``context/schema.sql``. This class initialises the
    database on first use; subsequent runs reuse the existing file.
    """

    def __init__(self, db_path: str | Path, schema_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._schema_path = (
            Path(schema_path)
            if schema_path
            else Path(__file__).parent / "schema.sql"
        )
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a SQLite connection that is closed cleanly on exit.

        ``sqlite3.Connection`` does not close itself inside a ``with``
        block — the context manager only commits or rolls back. On
        Windows, leaking connections prevents the database file from
        being deleted, which matters for tests. This wrapper both
        commits-or-rolls-back AND closes.
        """
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn, open(self._schema_path, encoding="utf-8") as f:
            conn.executescript(f.read())

    # -- writes --------------------------------------------------------

    def save_event(self, event: Event) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events
                    (id, source_system, entity_hint, observed_at, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event.id),
                    event.source_system,
                    event.entity_hint.value if event.entity_hint else None,
                    event.observed_at.isoformat(),
                    json.dumps(event.payload),
                    event.created_at.isoformat(),
                ),
            )

    def save_classification(
        self,
        classification: Classification,
        *,
        model_id: str,
        tier: str,
        agent_name: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
        latency_ms: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO classifications
                    (id, event_id, entity, category, priority, rationale, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(classification.id),
                    str(classification.event_id),
                    classification.entity.value,
                    classification.category.value,
                    classification.priority.value,
                    classification.rationale,
                    classification.created_at.isoformat(),
                ),
            )
            self._write_audit(
                conn,
                agent_name=agent_name,
                tier=tier,
                model_id=model_id,
                related_entity_table="classifications",
                related_entity_id=str(classification.id),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=estimated_cost_usd,
                latency_ms=latency_ms,
            )

    def save_analysis(
        self,
        analysis: Analysis,
        *,
        model_id: str,
        tier: str,
        agent_name: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
        latency_ms: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO analyses
                    (id, source_event_ids_json, involves_entities_json,
                     pattern_summary, recommended_action, escalate_to_l3, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(analysis.id),
                    json.dumps([str(i) for i in analysis.source_event_ids]),
                    json.dumps([e.value for e in analysis.involves_entities]),
                    analysis.pattern_summary,
                    analysis.recommended_action,
                    1 if analysis.escalate_to_l3 else 0,
                    analysis.created_at.isoformat(),
                ),
            )
            self._write_audit(
                conn,
                agent_name=agent_name,
                tier=tier,
                model_id=model_id,
                related_entity_table="analyses",
                related_entity_id=str(analysis.id),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=estimated_cost_usd,
                latency_ms=latency_ms,
            )

    def save_synthesis(
        self,
        synthesis: Synthesis,
        *,
        model_id: str,
        tier: str,
        agent_name: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
        latency_ms: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO syntheses
                    (id, analysis_ids_json, headline, briefing,
                     contradictions_json, suggested_next_watch_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(synthesis.id),
                    json.dumps([str(i) for i in synthesis.analysis_ids]),
                    synthesis.headline,
                    synthesis.briefing,
                    json.dumps(synthesis.contradictions_detected),
                    json.dumps(synthesis.suggested_next_watch),
                    synthesis.created_at.isoformat(),
                ),
            )
            self._write_audit(
                conn,
                agent_name=agent_name,
                tier=tier,
                model_id=model_id,
                related_entity_table="syntheses",
                related_entity_id=str(synthesis.id),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=estimated_cost_usd,
                latency_ms=latency_ms,
            )

    # -- reads ---------------------------------------------------------

    def recent_notable_classifications(self, *, limit: int = 50) -> list[Classification]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, event_id, entity, category, priority, rationale, created_at
                FROM classifications
                WHERE priority IN (?, ?)
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (EventPriority.NOTABLE.value, EventPriority.URGENT.value, limit),
            ).fetchall()
            return [self._row_to_classification(r) for r in rows]

    def recent_analyses(
        self,
        *,
        entities: list[EntityType] | None = None,
        limit: int = 20,
    ) -> list[Analysis]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        analyses = [self._row_to_analysis(r) for r in rows]
        if entities is None:
            return analyses

        wanted = set(entities)
        return [a for a in analyses if wanted.intersection(a.involves_entities)]

    # -- internal ------------------------------------------------------

    def _write_audit(
        self,
        conn: sqlite3.Connection,
        *,
        agent_name: str,
        tier: str,
        model_id: str,
        related_entity_table: str,
        related_entity_id: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
        latency_ms: int,
    ) -> None:
        conn.execute(
            """
            INSERT INTO audit_log
                (agent_name, tier, model_id, related_entity_table,
                 related_entity_id, input_tokens, output_tokens,
                 estimated_cost_usd, latency_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_name,
                tier,
                model_id,
                related_entity_table,
                related_entity_id,
                input_tokens,
                output_tokens,
                estimated_cost_usd,
                latency_ms,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    @staticmethod
    def _row_to_classification(row: sqlite3.Row) -> Classification:
        return Classification.model_construct(
            id=UUID(row["id"]),
            event_id=UUID(row["event_id"]),
            entity=EntityType(row["entity"]),
            category=_parse_category(row["category"]),
            priority=EventPriority(row["priority"]),
            rationale=row["rationale"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _row_to_analysis(row: sqlite3.Row) -> Analysis:
        data: dict[str, Any] = {
            "id": UUID(row["id"]),
            "source_event_ids": [UUID(x) for x in json.loads(row["source_event_ids_json"])],
            "involves_entities": [EntityType(x) for x in json.loads(row["involves_entities_json"])],
            "pattern_summary": row["pattern_summary"],
            "recommended_action": row["recommended_action"],
            "escalate_to_l3": bool(row["escalate_to_l3"]),
            "created_at": datetime.fromisoformat(row["created_at"]),
        }
        return Analysis.model_validate(data)


def _parse_category(raw: str):  # type: ignore[no-untyped-def]
    """Late import to avoid circular references during module load."""
    from schemas.models import EventCategory

    return EventCategory(raw)
