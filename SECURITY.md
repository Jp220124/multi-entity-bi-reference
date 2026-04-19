# Security Policy

## Reporting a Vulnerability

If you believe you have found a security issue in this repository,
please do **not** open a public issue. Email the maintainer directly
at the address listed on the LinkedIn profile in the README, or open
a private security advisory on GitHub:

> Repository → Security → Advisories → New draft security advisory

Expect an initial response within 3 business days.

## Scope

This is a reference implementation, not a production service. The
threat model and hardening expectations below apply to anyone who
vendors this code into their own deployment.

## What this repository does and does not protect against

### Protected

- **SQL injection.** Every call to `sqlite3.Connection.execute` uses
  parameterized placeholders. No string interpolation into SQL.
- **Schema-validated LLM output.** Every Claude response is parsed
  through a Pydantic model before it reaches the context layer.
  Malformed or out-of-enum output raises `ValidationError` and is
  never persisted. This bounds the blast radius of prompt-injection
  attacks: an attacker can at worst steer the model toward another
  value within the declared enum, not inject arbitrary content into
  downstream systems.
- **Atomic writes.** `save_classification`, `save_analysis`, and
  `save_synthesis` write the artifact and its `audit_log` row inside
  the same SQLite transaction. Partial states are not persisted.
- **No committed secrets.** `.env` is gitignored; `.env.example` is
  the only env file in the repo and contains no real keys. `git log`
  contains no API keys.
- **Idempotent event ingest.** `save_event` uses `INSERT OR IGNORE`
  so replaying the same event set is a no-op, not a crash.

### Not protected (consumer responsibility)

- **Prompt injection.** If your upstream `Event.payload` can be
  influenced by an attacker, they can attempt to redirect L1's
  classification. The Pydantic schema bounds the blast radius (only
  enum values can be written) but you should still validate and
  sanitize payloads before handing them to the orchestrator, and
  treat L1 output as untrusted.
- **Rate limiting / DoS against Anthropic.** The reference does not
  implement client-side rate limiting or circuit breakers. High-
  volume deployments should wrap the Anthropic client or use the
  Batch API.
- **API key leakage.** `ANTHROPIC_API_KEY` is read from the process
  environment. Operators must ensure it is not logged, not committed,
  and not exposed via process-environment-reading endpoints.
- **Multi-tenant isolation.** The SQLite backend is single-tenant.
  For multi-tenant deployments, swap `SQLiteContextStore` for a
  backend that enforces row-level security (e.g., Postgres/Supabase
  with RLS).
- **Supply chain.** Python dependencies are pinned to minimum
  versions in `pyproject.toml`, not exact versions. CI runs
  `pip-audit` on every push to surface known CVEs, but you should
  still maintain your own `pip freeze` or hash-pinned lockfile for
  production.

## Dependencies

Direct runtime dependencies are deliberately minimal:

- `anthropic` — Anthropic Claude SDK
- `pydantic` — schema validation at LLM boundaries
- `python-dotenv` — `.env` loader for local development

A full audit of indirect dependencies is run in CI (`.github/workflows/ci.yml`,
job `audit`) on every push.
