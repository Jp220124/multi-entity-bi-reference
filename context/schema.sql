-- Shared context layer schema.
-- Portable across SQLite and Postgres (Supabase) with trivial dialect
-- adjustments. Kept deliberately flat so audit queries stay simple.

-- Raw events entering the system, after any anonymization done by the
-- extractor. No PII should reach this table.
CREATE TABLE IF NOT EXISTS events (
    id              TEXT PRIMARY KEY,
    source_system   TEXT    NOT NULL,
    entity_hint     TEXT,
    observed_at     TEXT    NOT NULL,
    payload_json    TEXT    NOT NULL,
    created_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_observed_at ON events(observed_at);
CREATE INDEX IF NOT EXISTS idx_events_source      ON events(source_system);

-- L1 outputs: one classification per event.
CREATE TABLE IF NOT EXISTS classifications (
    id           TEXT PRIMARY KEY,
    event_id     TEXT NOT NULL REFERENCES events(id),
    entity       TEXT NOT NULL,
    category     TEXT NOT NULL,
    priority     TEXT NOT NULL,
    rationale    TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_classifications_entity   ON classifications(entity);
CREATE INDEX IF NOT EXISTS idx_classifications_priority ON classifications(priority);

-- L2 outputs: cross-entity analyses. Many-to-many with events is
-- modelled via the source_event_ids_json column (flat JSON array) so
-- the schema stays dialect-portable. A real deployment would normalize
-- into a join table.
CREATE TABLE IF NOT EXISTS analyses (
    id                      TEXT PRIMARY KEY,
    source_event_ids_json   TEXT NOT NULL,
    involves_entities_json  TEXT NOT NULL,
    pattern_summary         TEXT NOT NULL,
    recommended_action      TEXT,
    escalate_to_l3          INTEGER NOT NULL,   -- 0/1
    created_at              TEXT NOT NULL
);

-- L3 outputs: portfolio-level synthesis.
CREATE TABLE IF NOT EXISTS syntheses (
    id                        TEXT PRIMARY KEY,
    analysis_ids_json         TEXT NOT NULL,
    headline                  TEXT NOT NULL,
    briefing                  TEXT NOT NULL,
    contradictions_json       TEXT NOT NULL,
    suggested_next_watch_json TEXT NOT NULL,
    created_at                TEXT NOT NULL
);

-- Every tier writes to the audit log. Each row captures the agent that
-- ran, the tier, the model identifier, token usage, latency, and cost.
-- This is what enables the "no black box" guarantee: any output in the
-- system can be traced to the exact LLM call that produced it.
CREATE TABLE IF NOT EXISTS audit_log (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name             TEXT NOT NULL,
    tier                   TEXT NOT NULL,
    model_id               TEXT NOT NULL,
    related_entity_table   TEXT NOT NULL,  -- classifications / analyses / syntheses
    related_entity_id      TEXT NOT NULL,
    input_tokens           INTEGER NOT NULL,
    output_tokens          INTEGER NOT NULL,
    estimated_cost_usd     REAL NOT NULL,
    latency_ms             INTEGER NOT NULL,
    created_at             TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_log(agent_name);
CREATE INDEX IF NOT EXISTS idx_audit_tier  ON audit_log(tier);
