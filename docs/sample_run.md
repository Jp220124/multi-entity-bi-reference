# Sample End-to-End Run

This is a real output of `python examples/demo_run.py` against the live
Anthropic API. No edits, no curation — the raw briefing the system
produced for a synthetic portfolio with six events across retail,
hospitality, and real estate.

## Input

Six events in a single 24-hour window:

| Source system          | Entity      | Signal |
|---|---|---|
| `retail_pos`           | retail      | Location A — 58 tickets, $34,200 revenue, $589 avg basket |
| `retail_pos`           | retail      | Location B — 23 tickets, $9,450 revenue (routine) |
| `hotel_pms`            | hospitality | Property A — 14 open ops tickets, 11 opened in last 24h, 91.4% occupancy |
| `hotel_pms`            | hospitality | Property A — 7 late checkouts, 5 front-desk overrides |
| `weather_service`      | retail      | Storm advisory — moderate severity, ETA in ~2 hours, 8-hour duration |
| `building_maintenance` | real_estate | Building C — 9 open WOs, 6 HVAC, 3 after-hours |

## The synthesis the system produced

> **Headline:** Storm landfall in 2 hours meets portfolio-wide infrastructure strain across hospitality, real estate, and retail
>
> A single weather event is about to hit three entities that are already stressed. Hospitality is running high occupancy with a spike in maintenance tickets. Real estate shows elevated HVAC work orders and after-hours calls. Retail has a storm advisory with a two-hour ETA. Each signal looks routine alone. Together they point to one story: buildings and the people who fix them are overloaded right before the weather arrives.
>
> The risk is resource collision. The same HVAC techs, facilities vendors, and supervisors serve multiple properties. If demand peaks simultaneously, response times collapse and small failures become guest-facing, tenant-facing, and customer-facing incidents at once.
>
> Act in the next two hours. Convene an operations bridge across the three entities. Pre-position technicians at hospitality and real estate assets now, while roads are clear. Give hospitality front desks authority to resolve late-checkout friction without escalation. Confirm retail staffing and closure thresholds. Assign shared vendors to specific sites in priority order so they are not double-booked mid-storm.
>
> This is worth executive attention for the next 24 hours, primarily to ensure the three entities are not competing for the same people and parts.
>
> **Contradictions to note**
> - Entity-level dashboards read as normal seasonal variance; the portfolio view shows coordinated stress — single-entity monitoring is currently understating risk.
>
> **Watch list for next cycle**
> - HVAC and facilities ticket response times across hospitality and real estate during the storm window and 24 hours after
> - Shared vendor dispatch conflicts (same technician requested by two entities within the same hour)
> - Hospitality late-checkout volume and front-desk escalation rate during peak storm impact
> - Retail closure decisions vs. advisory timeline and any resulting staffing overtime
> - Post-storm work-order backlog by site to identify assets that absorbed hidden damage

## Run metrics

| Tier | Agent | Model | Calls | Input tokens | Output tokens | Cost | Latency |
|---|---|---|---:|---:|---:|---:|---:|
| L1 | `L1Classifier`  | `claude-haiku-4-5-20251001`           | 6 | 2,480 | 588   | $0.00543 | 10.3 s total |
| L2 | `L2Analyzer`    | `claude-sonnet-4-6`                   | 1 |   786 | 400   | $0.00836 | 8.5 s |
| L3 | `L3Synthesizer` | `claude-opus-4-7` (adaptive thinking) | 1 | 1,094 | 755   | $0.02434 | 13.0 s |
| **Total** |           |                                       | **8** | **4,360** | **1,743** | **$0.03812** | **31.8 s** |

Rates used: Haiku 4.5 ($1 / $5 per 1M tokens), Sonnet 4.6 ($3 / $15), Opus 4.7 ($5 / $25). Source: [Anthropic public pricing](https://docs.anthropic.com/en/docs/about-claude/pricing).

## What this demonstrates

1. **Cross-entity synthesis works.** L2 correctly identified that the
   storm advisory (external signal), the HVAC backlog (real estate),
   and the hospitality ops queue were part of a single resource-
   collision event, not three unrelated operational concerns.
2. **Contradictions are surfaced.** L3 flagged that entity-level
   dashboards would show normal variance while the portfolio view
   shows coordinated stress. A dashboard would never produce that
   finding.
3. **Every decision is auditable.** Eight rows in `audit_log`,
   traceable from the final synthesis all the way back to the raw
   events via `analysis_ids → source_event_ids → event_id`.
4. **Cost is predictable.** Under four cents for a full portfolio
   briefing across six events. At one cycle a day that is
   ~$12/month for the model layer.

## Reproducing

```bash
git clone https://github.com/Jp220124/multi-entity-bi-reference.git
cd multi-entity-bi-reference
pip install -e ".[dev]"
cp .env.example .env   # fill in ANTHROPIC_API_KEY
python examples/demo_run.py
```
