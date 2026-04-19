# Sample End-to-End Run

This is a real output of `python examples/demo_run.py` against the live
Anthropic API. No edits, no curation — the raw briefing the system
produced for a synthetic portfolio with six events across retail,
hospitality, and real estate.

## Input

Six events in a single 24-hour window:

| Source system | Entity | Signal |
|---|---|---|
| `jewelry_pos`         | retail        | Location A — 58 tickets, $34,200 revenue, $589 avg basket |
| `jewelry_pos`         | retail        | Location B — 23 tickets, $9,450 revenue (routine) |
| `hotel_pms`           | hospitality   | Property A — 14 open ops tickets, 11 opened in last 24h, 91.4% occupancy |
| `hotel_pms`           | hospitality   | Property A — 7 late checkouts, 5 front-desk overrides, 2 auto-fee applies |
| `port_authority_feed` | retail        | Cruise arrival — 3,100 expected pax, ETA 10:00 local |
| `building_maintenance`| real_estate   | Building C — 9 open WOs, 6 HVAC, 3 after-hours |

## The synthesis the system produced

> **Headline:** Cruise arrival of 3,100 hits hospitality at 91% occupancy with 11 open tickets; retail surge imminent
>
> One event matters today. A 3,100-passenger cruise ship is landing on us while hospitality is already running at 91.4% occupancy with 11 service tickets opened in the last 24 hours. The same wave will push retail foot traffic up within hours. We are absorbing peak demand with stretched staff.
>
> Hospitality is the weak point. Front desk is granting late checkout overrides under pressure, which compresses room turnover exactly when we need it fastest. Housekeeping cannot clear rooms if checkouts slip, and the ticket backlog will grow through the day unless we add hands now.
>
> Retail is the opportunity. Cruise passengers convert quickly and leave quickly. If checkout lines stall or key SKUs are not staged on the floor, we lose the sale and it does not come back tomorrow.
>
> Action is straightforward: surge staff the front desk and housekeeping, hold the line on late checkouts for the next 24 hours, and put retail floor managers on notice for a 1–4 hour traffic spike with full registers and stocked displays.
>
> **Contradictions to note**
> - Front desk is granting discretionary late checkout overrides at 91% occupancy, which contradicts the operational need to maximize room turnover during peak demand.
>
> **Watch list for next cycle**
> - Hospitality open service ticket count over next 24 hours
> - Late checkout override rate during cruise day
> - Room turnover time between checkout and next check-in
> - Retail checkout queue length and transaction throughput during 1–4 hour window after cruise disembarkation
> - High-demand retail SKU stockouts during cruise window

## Run metrics

| Tier | Agent | Model | Calls | Input tokens | Output tokens | Cost | Latency |
|---|---|---|---:|---:|---:|---:|---:|
| L1 | `L1Classifier`  | `claude-haiku-4-5-20251001` | 6 | 2,493 | 596   | $0.00139 | 9.1 s total |
| L2 | `L2Analyzer`    | `claude-sonnet-4-6`         | 1 |   796 | 391   | $0.00825 | 8.7 s |
| L3 | `L3Synthesizer` | `claude-opus-4-7` (adaptive) | 1 | 1,108 | 619   | $0.06305 | 11.1 s |
| **Total** |  |  | **8** | **4,397** | **1,606** | **$0.07267** | **28.8 s** |

## What this demonstrates

1. **Cross-entity synthesis works.** L2 correctly identified that the
   cruise arrival (retail entity, external signal) cascaded into
   hospitality pressure (operations category) — a pattern invisible
   looking at either business alone.
2. **Contradictions are surfaced.** L3 flagged the late-checkout
   override policy as contradicting the operational need for room
   turnover during peak demand — the kind of decision a dashboard
   would never produce on its own.
3. **Every decision is auditable.** Eight rows in `audit_log`,
   traceable from the final synthesis all the way back to the raw
   events via `analysis_ids → source_event_ids → event_id`.
4. **Cost is predictable.** Under eight cents for a full portfolio
   briefing across six events. At one cycle a day that is
   ~$26/month for the model layer alone.

## Reproducing

```bash
git clone https://github.com/Jp220124/multi-entity-bi-reference.git
cd multi-entity-bi-reference
pip install -e ".[dev]"
cp .env.example .env   # fill in ANTHROPIC_API_KEY
python examples/demo_run.py
```
