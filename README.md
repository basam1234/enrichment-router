![CI](https://github.com/basam1234/enrichment-router/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

# Enrichment Router

A **cost-and-latency-aware** data enrichment agent. Give it a company name (and optionally a domain), and it fills in missing fields — industry, country, public/private status, short description — by escalating through progressively more expensive data sources. It stops as soon as every field is resolved or the budget is exhausted.

---

## Architecture

The router uses a **three-tier escalation design** backed by a LangGraph cyclic state machine. Each tier is cheaper than the one above it. Fields resolved at a lower tier avoid the cost and latency of higher tiers entirely.

```
  ┌─────────────────────────────────────────────────────┐
  │                    try_tier                         │
  │   Runs the current tier's tool against unresolved   │
  │   fields in fields_needed.                          │
  └──────────────────────┬──────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │                check_sufficiency                    │
  │   Evaluates resolved fields against 0.5 threshold   │
  │   and budget headroom. Decides:                     │
  │   done / no-budget / no-more-tiers / escalate       │
  └──────────────────────┬──────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
       done /        escalate       no budget /
      no budget     (increment      no more tiers
                      tier, loop)       → END
```

The **budget tracker** enforces a joint constraint: both cost and latency must stay within budget on every tier. Budget checks use *pre-call declared estimates* — if a tool overshoots its estimate, the tracker records a drift event but does not retroactively reject the tier's results.

A separate **policy module** (`policy.py`) makes escalation decisions as a pure function — it receives field state and budget headroom as arguments rather than reaching into a tracker or resolver. This keeps decisions testable, replayable, and auditable from trace logs.

The **trace system** appends structured events at every decision point (tier entry, budget drift, policy outcome, finalization). The trace reducer uses LangGraph's `operator.add` (list concatenation), so each node appends events without stomping previous entries.

---

## Why LangGraph

The escalation flow is a **genuine cycle**: `try_tier → check_sufficiency → (increment tier, loop back)`, up to 3 iterations (tiers 0, 1, 2). This is not a DAG of branches — a linear chain would require unrolling every combination of "resolved after N tiers" into separate nodes or a hand-rolled while loop. LangGraph gives the cycle plus an append-only trace reducer for free, keeping the resolver a short declarative graph rather than imperative control flow.

LangGraph's `StateGraph` with `operator.add` reducer means each node can append trace events without coordinating with other nodes — the framework handles the merge.

---

## Tiers

| Tier | Tool | Resolves | Confidence | Cost | Latency |
|:----:|------|----------|:----------:|:----:|:-------:|
| 0 | Heuristic rules | industry, country (sometimes) | 0.35 | $0 | 10 ms |
| 1 | Wikipedia REST | short_description, country, is_public, industry | 0.65 – 0.85 | $0 | 600 ms |
| 2 | LLM (Llama 3.1 8B) | all four | 0.90 | modeled | 2000 ms |

### Tier 0 — Heuristic

Pure-Python rules with zero network calls. Industry is guessed from name keywords (e.g., "Labs" → Technology, "Capital" → Finance). Country is guessed from ccTLD extraction (e.g., `.de` → Germany, `.jp` → Japan). Confidence is intentionally low (0.35) — useful as a fallback but never authoritative.

### Tier 1 — Wikipedia

Calls the [Wikipedia REST API summary endpoint](https://en.wikipedia.org/api/rest_v1/page/summary/) to get a curated `extract` and `description`. Before resolving any fields, `_is_company_page` validates that the page actually describes a company — disambiguation pages (e.g., "Meta", "Meow") are rejected and escalate cleanly to Tier 2.

- **country**: substring-matched against a known country-name list
- **is_public**: keyword-matched against phrases like "publicly traded" or "privately held"
- **industry**: extracted via a strict `INDUSTRY_MAP` — known keywords like "financial services" or "software" resolve at 0.65; unknown keywords leave the field open for Tier 2
- **short_description**: Wikipedia's summary `description` field at 0.85

### Tier 2 — LLM

Sends the company name and domain to **Llama 3.1 8B Instant** via Groq's OpenAI-compatible API with a constrained prompt. Only *currently unresolved* fields are included — accepted fields from lower tiers are excluded, saving tokens. The LLM responds with a JSON object; `null` for unknown fields. Confidence is 0.90 (not 1.0, acknowledging hallucination risk).

---

## Escalation Policy & Confidence Thresholds

The policy module (`policy.py`) applies an **acceptance threshold of 0.5**. Fields below 0.5 are treated as unresolved, forcing escalation even though a lower tier "returned something."

This threshold was deliberately lowered from 0.6 to 0.5 — a **business rule prioritizing cost savings** over strict LLM verification. At 0.5, Wikipedia's `country` (0.65) and `is_public` (0.65) pass the gate and avoid expensive Tier 2 calls. Tier 0's 0.35 guesses still fall below, so they never block escalation on their own.

Four cases, evaluated in order:

| Priority | Decision | Condition |
|:--------:|----------|-----------|
| 1 | **Done** | All needed fields ≥ 0.5 |
| 2 | **Stop (budget)** | Unresolved fields remain + no budget for next tier |
| 3 | **Stop (no more tiers)** | Unresolved fields + budget exists + tier 2 is max |
| 4 | **Escalate** | Unresolved fields + budget + next tier exists |

---

## Modeled Cost

A free provider (Groq, via `llama-3.1-8b-instant`) is used for development and testing, while the system **reports what production against Claude Haiku 4.5 would cost**.

| | Input | Output |
|---|---|---|
| Claude Haiku 4.5 | $1.00 / MTok | $5.00 / MTok |

Rates sourced from [Anthropic's pricing page](https://platform.claude.com/docs/en/about-claude/pricing) — verify before real budgeting, as updates occur periodically.

---

## Recent Optimizations

### Llama 3.1 switch
Originally configured with `openai/gpt-oss-20b`, a reasoning model that incurs hidden chain-of-thought token overhead. Switching to `llama-3.1-8b-instant` eliminated this bloat, making the modeled cost a realistic Claude Haiku 4.5 proxy.

### Wikipedia company validation
The `_is_company_page` check rejects disambiguation and generic-concept pages. When a company name is also a common word (e.g., "Meta", "Meow"), the router escalates cleanly rather than polluting enrichment with "Topics referred to by the same term."

### Wikipedia industry extraction
Tier 1 resolves `industry` via a strict `INDUSTRY_MAP`. Known keywords → 0.65 confidence. Unknown → field stays open for Tier 2. This replaced an earlier short-circuit that unconditionally skipped Wikipedia for industry queries.

### LLM token optimization
Tier 2 constructs its prompt from *currently unresolved* fields only — not the full `fields_needed` set. Accepted fields are excluded, reducing both prompt and completion token counts.

### Database session injection
FastAPI uses `Depends(get_db)` to inject SQLAlchemy `Session` instances. Transaction control (`db.commit()`) lives at the API layer, not deep inside repository functions.

### LLM retries
The `GroqLLMClient` uses `tenacity` with exponential backoff (3 attempts, 2–10s) to handle Groq rate-limits (HTTP 429).

---

## SQLite Schema

Three tables, SQLAlchemy 2.0 ORM:

```
 records                  enrichment_runs               trace_events
 ┌──────────────┐       ┌───────────────────┐         ┌──────────────┐
 │ id (PK)      │◄──────│ record_id (FK)    │◄────────│ run_id (FK)  │
 │ name         │       │ id (PK)           │         │ id (PK)      │
 │ domain       │       │ status            │         │ node         │
 │ request_json │       │ total_cost_usd    │         │ detail (JSON)│
 │ created_at   │       │ total_latency_ms  │         │ created_at   │
 └──────────────┘       │ resolved_json     │         └──────────────┘
                        │ created_at        │
                        └───────────────────┘
```

**Why JSON for resolved fields?** The set of resolved fields varies per run. A normalized schema with nullable columns per field would be fragile — adding a field requires a migration. A single JSON column is schema-flexible and self-describing.

**Record deduplication:** Same name + domain → reuse record, accumulate runs. Different domains (e.g., `stripe.com` vs `stripe.jp`) → separate records.

---

## API

All endpoints under `/api`. Frontend served same-origin at `/`.

| Method | Path | Description |
|:------:|------|-------------|
| `GET` | `/` | Frontend (`static/index.html`) |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/records` | Submit a company for enrichment |
| `GET` | `/api/records` | List all past enrichment records |
| `GET` | `/api/records/{id}` | Single record detail |
| `GET` | `/api/records/{id}/trace` | Trace events for latest run |

### POST /api/records

```json
{
  "name": "Stripe",
  "domain": "stripe.com",
  "industry": null,
  "country": null,
  "is_public": null,
  "short_description": null,
  "max_cost_usd": 0.01,
  "max_latency_ms": 5000.0
}
```

- `null` or omitted → "needs enrichment"
- Explicit value (e.g., `"is_public": false`) → "caller already knows — skip"

---

## Frontend

A single self-contained HTML file at `src/enrichment_router/static/index.html` — embedded CSS, vanilla JavaScript. No build step, no npm, no framework.

The form collects company details and budget constraints, submits via `fetch()` to `/api/records`, and renders results in a table showing each resolved field's tier, confidence, and cost. A past-runs section lists all records with expandable trace views.

The critical `is_public` conversion: the `<select>` has three options — "Not specified" (omits key), "Yes" (`true`), "No" (`false`). Omitting means "needs enrichment"; `false` means "caller asserts private." Defaulting missing to false would break this distinction.

---

## Eval Harness

Compares the router against an **"always-LLM" baseline** over an 18-record synthetic dataset (`eval/companies.json`):

| Category | Count | Examples |
|----------|:-----:|----------|
| Real companies | 7 | Stripe, Notion, Figma, GitHub, Vercel |
| Fictional | 6 | Quantum Pie Holdings, Zxy Corp |
| Keyword-bearing | 4 | OpenAI Labs, Sequoia Capital |
| All-fields-supplied | 1 | Pre-Filled Corp |

The baseline skips tiers 0–1 and calls the LLM for every field, providing a direct cost/latency comparison. The eval generates a `StrategyMetrics` report and a side-by-side chart (`assets/cost_savings.png`).

```bash
python -m eval.run_eval
```

---

## Setup & Running

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com) (free tier works for development)

### Quick start

```bash
git clone https://github.com/basam1234/enrichment-router.git
cd enrichment-router
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env           # set GROQ_API_KEY
```

### Run the server

```bash
uvicorn enrichment_router.api.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) — the frontend is served at the root.

---

## Testing

```bash
pytest -q
```

The test suite uses **zero real network calls**:

- **FakeLLMClient** — scripted responses mapped to company names
- **Stub wiki fetcher** — injectable canned Wikipedia responses
- **In-memory SQLite** — `sqlite:///:memory:` with `PRAGMA foreign_keys = ON`
- **FastAPI TestClient** — full HTTP stack, no running server

CI: `pytest`, `ruff`, `black --check` on every push and PR via GitHub Actions.

---

## Security

- **No CORS** — frontend is same-origin; permissive CORS would be a regression
- **No hardcoded keys** — `GROQ_API_KEY` and `DATABASE_URL` from environment only
- `.env` is in `.gitignore`; `.env.example` committed with placeholders
- SQLite database is local — no network exposure
- LLM prompts constructed server-side — frontend never sends raw prompts

---

## Limitations & Future Work

- **Country matching** handles substring overlaps (e.g., "Equatorial Guinea" vs "Guinea") but is heuristic — unusual cases may produce false ambiguity
- **Real Claude integration** is a config change away (`LLMProviderConfig`), not built
- **Eval dataset is 18 records** — small, hand-curated, not statistically representative
- **Wikipedia rate limits** not modeled
- **Budget uses pre-call estimates** — a slow call can overrun actual budget; drift events are logged but don't retroactively reject
- **Trace ordering** by insertion id — correct for single-threaded, not concurrent runs
- **`is_public` keyword matching** in Tier 1 can miss reworded descriptions; 0.65 confidence reflects this
- **`INDUSTRY_MAP`** is a static lookup — novel or emerging industries require updates
- **No auth or rate limiting** on the API
- **Blocking enrichment** — the LangGraph run blocks the request thread

---

## License

MIT — see [LICENSE](LICENSE) for details.
