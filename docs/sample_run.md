# Sample End-to-End Run

This is a real output of `python examples/demo_run.py` against the live
Anthropic API. No edits, no curation — the raw briefing the system
produced for a synthetic portfolio with six events across retail,
hospitality, and real estate.

## Input

Six events in a single 24-hour window:

| Source system          | Entity      | Signal |
|---|---|---|
| `jewelry_pos`          | retail      | Location A — 58 tickets, $34,200 revenue, $589 avg basket |
| `jewelry_pos`          | retail      | Location B — 23 tickets, $9,450 revenue (routine) |
| `hotel_pms`            | hospitality | Property A — 14 open ops tickets, 11 opened in last 24h, 91.4% occupancy |
| `hotel_pms`            | hospitality | Property A — 7 late checkouts, 5 front-desk overrides |
| `port_authority_feed`  | retail      | Cruise arrival — 3,100 expected pax, ETA 10:00 local |
| `building_maintenance` | real_estate | Building C — 9 open WOs, 6 HVAC, 3 after-hours |

## The synthesis the system produced

> **Headline:** Cruise-day demand spike collides with HVAC backlog and 91% hotel occupancy — shared capacity at risk today
>
> Three entities are straining in the same 24-hour window, and the strains are connected. Hospitality is at 91.4% occupancy with open operational tickets and late-checkout policy breaking down. Retail faces a 3,100-passenger cruise arrival. Real estate is carrying a backlog of six HVAC issues with after-hours calls. Each looks routine alone. Together they are a single capacity event.
>
> The binding constraint is facilities. HVAC failures degrade the exact spaces where hotel guests sleep and where cruise shoppers browse. If climate control falters during peak load, guest complaints and walkouts will follow within hours, not days. Maintenance staff are the scarce resource, and they are being pulled into reactive work instead of pre-positioned for the peak.
>
> The second constraint is front-line labor. Hospitality's inability to enforce late checkout signals the front desk is already overloaded. Retail has not yet absorbed the cruise wave. Staff cannot be borrowed across entities if each is firefighting its own queue.
>
> Action this cycle is operational, not strategic: a same-day triage call between the hospitality GM, retail manager, and facilities lead to sequence HVAC repairs toward guest rooms and retail floor first, enforce late-checkout discipline, and pre-stage retail coverage before the ship docks. This is worth executive visibility because the failure mode is a visible guest-experience incident on a high-revenue day, not a slow drift.
>
> **Contradictions to note**
> - Late-checkout policy is documented but not being enforced, contradicting the stated operating standard for room turnover on high-occupancy days
> - Facilities is treated as a background function, yet today it is the binding constraint on both hospitality and retail revenue
>
> **Watch list for next cycle**
> - HVAC ticket count and time-to-close over the next 48 hours, segmented by guest-facing vs back-of-house zones
> - Hotel late-checkout compliance rate on cruise days
> - Retail sales per labor hour during the 3,100-passenger arrival window
> - Guest complaint volume tagged to room temperature or air quality
> - After-hours maintenance call frequency as a leading indicator of deferred-maintenance debt

## Run metrics

| Tier | Agent | Model | Calls | Input tokens | Output tokens | Cost | Latency |
|---|---|---|---:|---:|---:|---:|---:|
| L1 | `L1Classifier`  | `claude-haiku-4-5-20251001`           | 6 | 2,486 | 582   | $0.00135 | 11.2 s total |
| L2 | `L2Analyzer`    | `claude-sonnet-4-6`                   | 1 |   789 | 473   | $0.00946 | 14.2 s |
| L3 | `L3Synthesizer` | `claude-opus-4-7` (adaptive thinking) | 1 | 1,176 | 821   | $0.07921 | 16.0 s |
| **Total** |           |                                       | **8** | **4,451** | **1,876** | **$0.09003** | **41.3 s** |

## What this demonstrates

1. **Cross-entity synthesis works.** L2 correctly identified that the
   cruise arrival (retail entity, external signal), the HVAC backlog
   (real estate), and the hospitality ops queue were part of a single
   capacity event, not three unrelated operational concerns.
2. **Contradictions are surfaced.** L3 flagged two: the late-checkout
   policy breakdown and the treatment of facilities as a background
   function when it is actually the binding constraint. A dashboard
   would never produce either finding.
3. **Every decision is auditable.** Eight rows in `audit_log`,
   traceable from the final synthesis all the way back to the raw
   events via `analysis_ids → source_event_ids → event_id`.
4. **Cost is predictable.** Under ten cents for a full portfolio
   briefing across six events. At one cycle a day that is
   ~$27/month for the model layer.

## Reproducing

```bash
git clone https://github.com/Jp220124/multi-entity-bi-reference.git
cd multi-entity-bi-reference
pip install -e ".[dev]"
cp .env.example .env   # fill in ANTHROPIC_API_KEY
python examples/demo_run.py
```
