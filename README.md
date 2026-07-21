![CI](https://github.com/basam1234/enrichment-router/actions/workflows/ci.yml/badge.svg)

# Enrichment Router

A cost-and-latency-aware data enrichment agent. Given a company name (and optionally a domain), the router fills in missing fields — industry, country, public/private status, short description — by escalating through progressively more expensive data sources, stopping as soon as every field is resolved or the budget is exhausted.

## Table of Contents

- [Architecture](#architecture)
- [Why LangGraph](#why-langgraph)
- [Tiers](#tiers)
- [Escalation Policy & Confidence Thresholds](#escalation-policy--confidence-thresholds)
- [Modeled Cost](#modeled-cost)
- [Recent Optimizations](#recent-optimizations)
- [SQLite Schema](#sqlite-schema)
- [API](#api)
- [Frontend](#frontend)
- [Eval Harness](#eval-harness)
- [Setup & Running](#setup--running)
- [Testing](#testing)
- [Security](#security)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)

## Architecture

The router uses a **three-tier escalation design** backed by a LangGraph cyclic state machine. Each tier is cheaper than the one above it. Fields resolved at a lower tier avoid the cost and latency of higher tiers entirely.

```
  ┌────────────────────────────────────────────────────┐
  │                   try_tier                         │
  │  Runs the current tier's tool against unresolved   │
  │  fields in fields_needed.                          │
  └──────────┬─────────────────────────────────────────┘
             │
             ▼
  ┌────────────────────────────────────────────────────┐
  │               check_sufficiency                    │
  │  Evaluates resolved fields against the acceptance  │
  │  threshold (0.5) and budget headroom. Decides:     │
  │  done / no-budget / no-more-tiers / escalate.      │
  └──────────┬─────────────────────────────────────────┘
             │
             ├── done / budget / no-more-tiers ──► END
             │
             └── escalate ──► (increment tier, loop to try_tier)
```

The **budget tracker** enforces a joint constraint: both cost and latency must stay within budget on every tier. Budget checks use *pre-call declared estimates* — if a tool overshoots its estimate, the tracker records a drift event but does not retroactively reject the tier's results.

A separate **policy module** (`policy.py`) makes escalation decisions as a pure function — it receives field state and budget headroom as arguments rather than reaching into a tracker or resolver. This keeps decisions testable, replayable, and auditable from trace logs.

The **trace system** appends structured events at every decision point (tier entry, budget drift, policy outcome, finalization). The trace reducer uses LangGraph's `operator.add` (list concatenation), so each node appends events without stomping previous entries.

## Why LangGraph

The escalation flow is a **genuine cycle**: `try_tier → check_sufficiency → (increment tier, loop back)`, up to 3 iterations (tiers 0, 1, 2). This is not a DAG of branches — a linear chain would require unrolling every combination of "resolved after N tiers" into separate nodes or a hand-rolled while loop. LangGraph gives the cycle plus an append-only trace reducer for free, keeping the resolver a short declarative graph rather than imperative control flow.

LangGraph's `StateGraph` with `operator.add` reducer means each node can append trace events without coordinating with other nodes — the framework handles the merge.

## Tiers

| Tier | Tool | Resolves | Confidence | Cost | Latency (declared) |
|------|------|----------|------------|------|---------------------|
| 0 | Heuristic rules | industry, country (sometimes) | 0.35 | $0 | 10 ms |
| 1 | Wikipedia REST | short_description, country, is_public, industry | 0.65 – 0.85 | $0 | 600 ms |
| 2 | LLM (Llama 3.1 8B via Groq) | all four | 0.9 | modeled | 2000 ms |

**Tier 0 (Heuristic):** Pure-Python rules with zero network calls. Industry is guessed from name keywords (e.g., "Labs" → Technology, "Capital" → Finance). Country is guessed from ccTLD extraction (e.g., `.de` → Germany, `.jp` → Japan). Confidence is intentionally low (0.35) because these are educated guesses — useful as a fallback but never authoritative.

**Tier 1 (Wikipedia):** Calls the [Wikipedia REST API summary endpoint](https://en.wikipedia.org/api/rest_v1/page/summary/) to get a curated `extract` and `description`. Before resolving any fields, `_is_company_page` validates that the page actually describes a company or organization — disambiguation pages (e.g., "Meta", "Meow") are rejected and treated as `not_found`, forcing clean escalation to Tier 2. Country is extracted by substring-matching against a known list of country names. `is_public` is determined by keyword-matching phrases like "publicly traded" or "privately held". Industry is extracted via a strict `INDUSTRY_MAP` — if a known keyword like "financial services" or "software" appears in the extract or description, the field is resolved at 0.65 confidence. If no keyword matches, the field remains unresolved, allowing escalation to Tier 2.

**Tier 2 (LLM):** Sends the company name and domain to Llama 3.1 8B Instant via Groq's OpenAI-compatible API with a constrained prompt. The graph only includes *currently unresolved* fields in the prompt — fields already resolved by lower tiers at or above the acceptance threshold are excluded, saving prompt and completion tokens. The LLM responds with a JSON object; `null` is used for unknown fields. Confidence is 0.9 — lower than 1.0 to acknowledge that LLMs can hallucinate.

## Escalation Policy & Confidence Thresholds

The policy module (`policy.py`) applies an **acceptance threshold of 0.5** to every resolved field. Fields with confidence below 0.5 are treated as unresolved, forcing the router to escalate to the next tier even though a lower tier "returned something."

This threshold was deliberately lowered from 0.6 to 0.5 — a business rule that prioritizes cost savings over strict LLM verification. At 0.5, Wikipedia's `country` (0.65) and `is_public` (0.65) resolutions pass the gate and avoid expensive Tier 2 escalation. Tier 0's 0.35-confidence guesses still fall below the threshold, so they never block escalation on their own.

Four cases are evaluated in order:

1. **Done** — all needed fields have accepted resolutions (confidence ≥ 0.5).
2. **Stop (budget)** — unresolved fields remain but no budget headroom for the next tier.
3. **Stop (no more tiers)** — unresolved fields remain, budget exists, but tier 2 is the max.
4. **Escalate** — unresolved fields remain, budget exists, next tier exists. Increment and loop.

## Modeled Cost

Cost was a real constraint — the router must demonstrate savings over "just ask the LLM for everything." A free provider (Groq, via `llama-3.1-8b-instant`) is used for development and testing, while the system **reports what production against Claude Haiku 4.5 would cost.**

Rates are $1.00/MTok input and $5.00/MTok output, sourced from <https://platform.claude.com/docs/en/about-claude/pricing>, and should be verified before relying on this for real budgeting, as Anthropic updates pricing periodically (e.g., Sonnet had a scheduled rate change on 2026-09-01).

## Recent Optimizations

**Llama 3.1 switch:** The LLM tier was originally configured with `openai/gpt-oss-20b`, a reasoning model that incurs hidden chain-of-thought token overhead even at low `reasoning_effort`. Switching to `llama-3.1-8b-instant` eliminated this bloat. The modeled cost is now a realistic upper bound for what Claude Haiku 4.5 would consume on the same prompt, rather than a Groq-specific inflation.

**Wikipedia company validation:** The `_is_company_page` check prevents Wikipedia disambiguation pages and generic concept pages from polluting enrichment results. When a company name is also a common word (e.g., "Meta", "Meow"), Wikipedia returns a disambiguation page with "Topics referred to by the same term" — these are rejected, and the router escalates cleanly to Tier 2.

**Wikipedia industry extraction:** Tier 1 now resolves `industry` via a strict `INDUSTRY_MAP` database. If a known keyword (e.g., "financial services", "software", "pharmaceutical") appears in the Wikipedia description or extract, the field is resolved at 0.65 confidence. If no keyword matches, the field remains unresolved, allowing Tier 2 to provide a more nuanced classification. This replaced an earlier short-circuit that unconditionally skipped Wikipedia for industry queries.

**LLM token optimization:** Tier 2 constructs its prompt from *currently unresolved* fields only — not from the full original `fields_needed` set. Fields already accepted by lower tiers (above the 0.5 threshold) are excluded, reducing both prompt and completion token usage.

**Database session injection:** The FastAPI API uses `Depends(get_db)` to inject SQLAlchemy `Session` instances into route handlers, rather than opening ad-hoc sessions per repository call. This gives the caller control over the transaction lifecycle — `db.commit()` happens at the API layer, not deep inside repository functions.

**LLM retries:** The `GroqLLMClient` uses `tenacity` with exponential backoff (up to 3 attempts, 2–10s wait) to handle Groq rate-limit responses (HTTP 429) gracefully.

## SQLite Schema

Three tables, all managed by SQLAlchemy ORM:

```
records                enrichment_runs              trace_events
┌──────────────┐      ┌──────────────────┐        ┌──────────────┐
│ id (PK)      │◄─────│ record_id (FK)   │◄───────│ run_id (FK)  │
│ name         │      │ id (PK)          │        │ id (PK)      │
│ domain       │      │ status           │        │ node         │
│ request_json │      │ total_cost_usd   │        │ detail (JSON)│
│ created_at   │      │ total_latency_ms │        │ created_at   │
└──────────────┘      │ resolved_json    │        └──────────────┘
                      │ created_at       │
                      └──────────────────┘
```

**Why resolved fields are JSON**: The set of resolved fields varies per run (different tiers resolve different subsets). A normalized schema with nullable columns for every possible field would be fragile — adding a new field requires a migration. A single JSON column is schema-flexible and self-describing. Each `ResolvedField` is flattened into a dict with keys `value`, `tier`, `confidence`, and `caller_supplied`, so the frontend can render results directly without needing the Pydantic model definitions.

Record deduplication: A company submitted multiple times with the same name and domain reuses the same record row, accumulating runs. Different domains (e.g., `stripe.com` vs `stripe.jp`) are separate records.

## API

All endpoints served by FastAPI under `/api`. The frontend is served same-origin at `/`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serve the frontend (`static/index.html`) |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/records` | Submit a company for enrichment |
| `GET` | `/api/records` | List all past enrichment records |
| `GET` | `/api/records/{id}` | Get a single record's full detail |
| `GET` | `/api/records/{id}/trace` | Get trace events for a record's latest run |

The `POST /api/records` request body:

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

Fields set to `null` or omitted mean "needs enrichment." Fields set to explicit values (e.g., `"is_public": false`) mean "caller already knows — don't enrich."

## Frontend

A single self-contained HTML file at `src/enrichment_router/static/index.html` with embedded CSS and vanilla JavaScript. No build step, no npm, no framework — the page is small enough that vanilla JS is clearer than a bundled framework.

The form collects company details and budget constraints, submits via `fetch()` to `/api/records`, and renders results in a table showing each resolved field, its tier, confidence, and cost. A past runs section lists all records with expandable trace views.

The critical `is_public` conversion: the `<select>` element has three options — "Not specified" (omits the key from the JSON body entirely), "Yes" (`true`), and "No" (`false`). Omitting the key means "needs enrichment," while `false` means "caller asserts this is a private company." Sending the string `"false"` or defaulting missing to false would break this distinction.

## Eval Harness

The eval harness (`eval/run_eval.py`) compares the router against an "always-LLM" baseline (`eval/baseline.py`) over an 18-record synthetic dataset (`eval/companies.json`).

The dataset covers four categories:

- **7 real companies** (Stripe, Notion, Airtable, Figma, Linear, GitHub, Vercel) — all with real domains
- **6 fictional companies** (Quantum Pie Holdings, Zxy Corp, etc.) — no domains or `.invalid` domains
- **4 keyword-bearing companies** (OpenAI Labs, Sequoia Capital, Sweet Bakery, Boston Robotics) — names that trigger tier 0 heuristic keywords
- **1 all-fields-supplied** (Pre-Filled Corp) — all four target fields set, so no work is needed

The baseline skips tiers 0 and 1 entirely, calling the LLM for every field. This provides a direct cost/latency comparison. The eval generates a `StrategyMetrics` report and a side-by-side bar chart (`assets/cost_savings.png`) comparing total modeled cost and total latency.

### Running the eval

```bash
eval/companies.json  # edit or replace as needed
python -m eval.run_eval
```

The chart is written to `assets/cost_savings.png`.

## Setup & Running

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com) (free tier works for development)

### Quick start

```bash
git clone https://github.com/basam1234/enrichment-router.git
cd enrichment-router
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env: set GROQ_API_KEY to your key
```

### Run the server

```bash
uvicorn enrichment_router.api.main:app --reload
```

Open `http://localhost:8000` — the frontend is served at the root.

### Run the eval

```bash
python -m eval.run_eval
```

## Testing

```bash
pytest -q
```

The test suite uses:

- **FakeLLMClient** — scripted LLM responses mapped to company names, no network calls
- **Stub wiki fetcher** — injectable canned Wikipedia responses, no real HTTP
- **In-memory SQLite** — `configure_engine("sqlite:///:memory:")` per test with `PRAGMA foreign_keys = ON`
- **FastAPI TestClient** — full HTTP stack without a running server

CI runs on every push and PR via GitHub Actions (`pytest`, `ruff`, `black --check`).

## Security

**No CORSMiddleware is added** — the frontend is served same-origin by the same FastAPI app, so cross-origin config is unnecessary and permissive CORS would be a security regression.

The project follows standard security practices:

- API keys come from environment variables only (`GROQ_API_KEY`, `DATABASE_URL`). No hardcoded credentials.
- `.env.example` is committed with placeholder values; `.env` is in `.gitignore`.
- The SQLite database is local — no network exposure of the data store.
- LLM prompts are constructed server-side — the frontend never sends raw prompts.

## Limitations & Future Work

- **Nested country name matching**: The Wikipedia extract scanner handles substring overlaps (e.g., "Equatorial Guinea" contains "Guinea") by keeping only the longest match, but this is a heuristic — deeply nested or unusual cases may still produce false ambiguity.
- **Real Claude integration** is a config change away (`LLMProviderConfig`), not built.
- The eval dataset is **18 records** — small, hand-curated, and not statistically representative.
- The Wikipedia REST API has **practical rate limits** not modeled here.
- **Budget check uses pre-call declared estimates**; a single unexpectedly slow call (e.g., network jitter on Wikipedia or an LLM spike) can overrun the actual budget. Drift events are logged but do not retroactively reject the tier.
- **Trace events are ordered by insertion id** — correct for single-threaded runs only. Concurrent enrichment runs could interleave trace events across records.
- The `is_public` determination in tier 1 uses simple keyword matching against article text, which can miss reworded descriptions or match stale text. The 0.65 confidence reflects this.
- The `INDUSTRY_MAP` is a static lookup table — it will miss novel or emerging industries and relies on exact keyword matches in the description/extract text.
- No authentication or rate limiting on the API.
- No async enrichment — the LangGraph run blocks the request thread.

## License

MIT — see [LICENSE](LICENSE) for details.
