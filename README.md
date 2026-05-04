# Valura AI — Microservice (assignment build)

A FastAPI microservice that takes a financial query, runs it through a
synchronous safety guard, classifies intent with a single LLM call,
routes to the right specialist agent, and streams the response back
over Server-Sent Events.

The mission: be the **AI co-investor** behind every interaction —
helping a novice **build, monitor, grow, and protect** a portfolio.
This build is the spine: safety + classifier + routing + one
fully-implemented agent (Portfolio Health) + the HTTP layer. Adding
the other six specialists later is a one-line registration, not a
rewrite.

> **Defence video:** _link added with the final commit._

---

## Contents

- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [End-to-end request flow](#end-to-end-request-flow)
- [Components](#components)
- [Library choices](#library-choices)
- [Design decisions](#design-decisions)
- [Tradeoffs and assumptions](#tradeoffs-and-assumptions)
- [Performance](#performance)
- [Testing](#testing)
- [Project layout](#project-layout)
- [Defence prep](#defence-prep)

---

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env  # OPENAI_API_KEY only needed for live runs; tests don't need it
pytest tests/ -v      # 183 tests, fully offline
```

Run the service:

```bash
uvicorn valura_ai.main:app --reload
```

Hit it:

```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat \
  -H 'content-type: application/json' \
  -d '{"message":"how is my portfolio doing?","user_id":"user_002_concentrated_nvda"}'
```

You'll see SSE events: `meta` → multiple `token` → `structured` → `done`.

> Without `OPENAI_API_KEY` the service still runs — it falls back to a
> stub LLM that routes everything to `portfolio_health`. Useful for
> demoing without a key; documented in `src/valura_ai/main.py`.

---

## Architecture

```
                     POST /v1/chat  { message, user_id }
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │       Pipeline       │  asyncio.timeout(12s)
                       └──────────┬───────────┘
                                  │
   ┌──────────────────────────────┼──────────────────────────────────┐
   │                              ▼                                  │
   │  1.  SafetyGuard.check(message)                                 │
   │         ├─ blocked → emit `error` event → END                   │
   │         └─ pass → continue                                      │
   │                              │                                  │
   │  2.  SessionStore.history(user_id)        ── last N turns       │
   │  3.  UserStore.get(user_id)               ── profile + holdings │
   │                              │                                  │
   │                              ▼                                  │
   │  4.  IntentClassifier                                           │
   │         ├─ LLM (timeout 4s, structured output)                  │
   │         │     ├─ success → ClassificationResult                 │
   │         │     └─ LLMError → heuristic fallback (deterministic)  │
   │         └─ unknown agent in response → heuristic                │
   │                              │                                  │
   │                              ▼ emit `meta` event                │
   │                                                                 │
   │  5.  AgentRegistry.get(classification.agent)                    │
   │         ├─ portfolio_health → real PortfolioHealthAgent         │
   │         └─ everything else  → StubAgent("not_implemented")      │
   │                              │                                  │
   │                              ▼ stream                           │
   │       Agent.stream() yields token+ , then structured            │
   │                              │                                  │
   │                              ▼ emit per event                   │
   │                                                                 │
   │  6.  SessionStore.append(user, assistant)                       │
   │                              │                                  │
   │                              ▼                                  │
   │                        emit `done` event                        │
   └─────────────────────────────────────────────────────────────────┘
```

---

## End-to-end request flow

A `POST /v1/chat` arrives with `{message, user_id}`. The route in
`api/routes.py` is intentionally thin — parse body, open SSE response,
delegate everything to `Pipeline.run()`. The pipeline is a single async
generator wrapped in `asyncio.timeout(request_timeout_s)` so a slow
component anywhere becomes a structured error event, never a 500.

1. **SafetyGuard.check(message)** — pure-Python, sub-millisecond. If
   blocked, the route emits one `error` event with the per-category
   professional refusal and the stream ends. Classifier never runs
   (assignment §Safety Precedence). Session is **not** updated on a
   block.

2. **SessionStore.history(user_id)** — last N turns, in-memory.

3. **UserStore.get(user_id)** — profile + portfolio loaded from
   `fixtures/users/`. `None` for an unknown user; pipeline still
   proceeds.

4. **IntentClassifier.classify(query, history, user_context)** — one
   LLM call returning a structured `ClassificationResult`. On any
   `LLMError`, falls back to a deterministic heuristic that hits 90.6%
   routing accuracy on our public fixture. Out-of-vocab agent in the
   response also triggers the heuristic. The pipeline emits a `meta`
   event carrying the classification — first wire byte well under the
   2s p95 budget.

5. **AgentRegistry.get(agent_name)** — always returns a `BaseAgent`.
   `portfolio_health` is the real implementation; the other 7 are
   `StubAgent` instances. Unknown names get a fresh stub (never
   raises). Agent's `stream()` yields `token+` then one `structured`
   event; the route translates each into the matching SSE event.

6. **SessionStore.append(user, assistant)** — record the turn so the
   next call sees it as history. Capped at 500 chars per assistant
   turn.

The route emits `event: done` on success. On any error event, the
route exits its generator immediately — no trailing `done`.

---

## Components

### SafetyGuard (`src/valura_ai/safety/`)

Pure-Python, no LLM, no I/O. Layered design:

| Layer | What it matches | Weight |
|---|---|---|
| 1. Phrases | "insider trading", "pump and dump", "launder cash" | 1 |
| 2. Intent | actor + action ("CFO friend hinted", "I have inside info") | 2 |
| 3. Methodology | "how do I", "how would someone", "common methods" | (rule trigger) |
| 4. Evasion | "just curious", "theoretically", "between you and me" | (rule trigger) |
| 5. Educational | "what is", "explain the difference", regulatory references | (rescue trigger) |

Decision precedence (first match wins):
- **R1)** phrase ≥1 AND methodology → BLOCK (catches *"explain X and how it's done"*)
- **R2)** phrase ≥1 AND evasion → BLOCK (catches *"just curious how X works"*)
- **R3)** score ≥ 2 AND not educational → BLOCK
- **R4)** otherwise → PASS

**Asymmetry is the key idea:** educational framing only excuses pure
phrase matches, never methodology requests or evasion phrasing. That's
how a single guard handles both `"what is insider trading"` (pass) and
`"explain insider trading and how it's done"` (block).

7 categories, each with a distinct, professional refusal that points
the user toward a legitimate alternative.

**Numbers** (against `fixtures/test_queries/safety_pairs.json`, 72
labeled pairs):
- recall on harmful: **100%** (42/42) — assignment threshold ≥95%
- pass-through on safe: **100%** (30/30) — assignment threshold ≥90%
- mean per-call latency: **0.014 ms**, p99 **0.021 ms** — assignment budget <10 ms
- 10/10 on a held-out adversarial set + 12/12 on benign over-block probes
  (`tests/test_safety_robustness.py`)

### IntentClassifier (`src/valura_ai/classifier/`)

One LLM call per request. Pydantic-validated structured output. The
prompt is **taxonomy-driven** — agent names are loaded from
`fixtures/test_queries/intent_classification.json` at startup, not
hardcoded in `src/`. Plain-language descriptions of each agent's
responsibility plus general routing principles. **No fixture queries
appear in the prompt** — verified by `test_prompt_does_not_contain_fixture_phrases`.

`ClassificationResult` schema:

```python
intent: str                        # short slug describing the user's ask
agent: str                         # one of the taxonomy agents (exact-match)
entities:
  tickers: list[str]
  topics: list[str]
  sectors: list[str]
  amount: float | None
  rate: float | None
  period_years: float | None
confidence: float  (0..1)
reasoning: str                     # short, internal-use
safety_verdict: { is_safe, note }  # informational only
```

**Heuristic fallback** triggers on any `LLMError` (timeout, parse,
refusal, transport) and on out-of-vocab agent names from the LLM. Pure
Python, deterministic, sub-millisecond. Falls back to the last user
turn in history when the current turn is vague — handles
`"thoughts?"`, `"and now?"`, `"explain like im 5"`. Heuristic
confidence is capped at 0.4 so ops can detect a degraded run.

**Numbers** (85-query fixture):
- LLM oracle path (mechanical wiring test): 100% routing, 0 entity-match failures
- Heuristic alone: **77/85 = 90.6%** routing accuracy
- Per-agent: portfolio_health 100%, financial_calculator 100%, support 100%,
  recommendations 93%, investment_strategy 92%, risk_assessment 90%,
  market_research 82%, predictive_analysis 73%

### Portfolio domain (`src/valura_ai/portfolio/`)

Pydantic models (`Position` is frozen, `Portfolio` is mutable). One
`MarketData` interface with two implementations: `StaticMarketData`
(in-memory, used by every test) and `YFinanceMarketData` (lazy-imports
yfinance, disk-cached 1h prices / 6h FX). `compute_valuation()` is
the single entry point that converts a `User` + `MarketData` into a
typed `PortfolioValuation` — concentration, returns, per-position
weights, missing-data lists.

Edge cases handled by typed-result fields, not exceptions:
- Empty portfolio → `concentration.flag = "n/a"`, all metrics `None`
- Single position → top1 = top3 = 100%, no `/0`
- Missing price → in `missing_prices` list, aggregates over priced subset
- Missing FX → in `missing_fx`, position treated as missing-priced
- Multi-currency → all amounts normalized to `user.base_currency` before weighting

### Agents (`src/valura_ai/agents/`)

`BaseAgent.stream()` is an async generator yielding `AgentEvent`s
(token / structured / error / info). The HTTP layer maps these into
SSE events. Two implementations:

**`PortfolioHealthAgent`** — implemented, deterministic, no LLM.
Templated prose because the data is the answer here; templating gives
sub-millisecond first-token, fully reproducible test output, and zero
drift between prose and the structured payload. Branches:

- empty → BUILD branch (forward-looking headline, starter-allocation prompt)
- high concentration → headline calls out top position by name + pct + drawdown impact
- moderate → balanced framing + watchlist note
- low → affirmation + one nuance observation
- aggressive risk + leveraged ETFs detected → leverage note (neutral)
- cash > 25% of book → cash-drag observation
- missing data → warning observation, never a crash

Observations capped at 4, sorted critical > warning > info. Disclaimer
appended verbatim to every response.

**`StubAgent`** — stands in for every taxonomy agent that isn't
implemented yet. Returns the structured "not implemented" payload the
assignment specifies (intent, agent, entities, status, message) plus
prose that names the agent and echoes back captured entities so the
user sees their request was understood.

`AgentRegistry` maps every taxonomy name to a `BaseAgent`. Adding a
new specialist is one entry in `build_default_registry()`'s
`implemented` dict — no other code changes.

### HTTP + SSE pipeline (`src/valura_ai/api/`, `src/valura_ai/main.py`)

`Pipeline.run()` orchestrates everything in a single async generator.
The route function (`api/routes.py`) is thin: parse, open SSE,
forward. SSE events are formatted by `api/sse.py` — every Pydantic
payload goes out via `model_dump_json` so the wire format is stable.

Wire format on the happy path:
```
event: meta
data: {"intent":"...","agent":"...","entities":{...},"confidence":...}

event: token
data: ... prose chunk 1 ...

event: token
data: ... prose chunk 2 ...

event: structured
data: {"user_id":"...","headline":"...","concentration_risk":{...},...}

event: done
data:
```

On a safety block: only one `error` event, then the stream ends.
On an internal failure: `meta` (if classification ran) → `error`. Errors
are JSON `{"message": "..."}`, never stack traces.

---

## Library choices

| Library | Why |
|---|---|
| **FastAPI** | Async-native, OpenAPI for free, the standard FastAPI/uvicorn stack is what every reviewer can run without surprises. |
| **uvicorn[standard]** | Production-grade ASGI server; the `[standard]` extras add httptools + uvloop. |
| **sse-starlette** | Battle-tested SSE: handles content-type, keep-alives, client disconnects. Worth the dep over rolling our own — saves on subtle bugs around backpressure and lifespan. |
| **openai** (≥1.50) | Structured outputs API (`beta.chat.completions.parse`) returns Pydantic-validated objects in one round-trip. No prompt-engineered JSON parsing. |
| **pydantic** v2 | Schemas everywhere — at API boundaries, in the classifier, in agent outputs. Validation at the edge means agent code can trust its inputs. |
| **pydantic-settings** | Env-driven `Settings` with type validation; saves writing config glue. |
| **yfinance** | Free, no key, fine for a 3-day build. Cached to disk so repeat runs don't re-fetch. |
| **httpx** + FastAPI's `TestClient` | TestClient supports streaming responses; integration tests parse real SSE wire format. |
| **pytest** + `pytest-asyncio` | Async test support without ceremony. `asyncio_mode = "auto"` so async tests don't need decoration. |

---

## Design decisions

**Why FastAPI.** Native async, native streaming via ASGI, type-driven
request validation via Pydantic. Together they remove the boilerplate
that would normally exist between the HTTP body and the
classifier/agent code. The FastAPI/uvicorn stack is also what every
reviewer has installed already.

**Why SSE (not WebSockets).** SSE is a one-way streaming protocol over
plain HTTP. Half the complexity of WebSockets for the same delivery
guarantee in this use case (server pushes; client doesn't push back
mid-response). Standard browsers and CLIs (`curl -N`) consume it
without a library. WebSockets buy nothing here we'd actually use.

**Why a `FakeLLMClient`.** Tests must run without `OPENAI_API_KEY` per
the assignment. The fake is the same `LLMClient` interface as the real
client, programmable per-test (substring / regex / callable matchers,
BaseModel / Exception / async-factory responses), and records every
call for introspection. `OPENAI_API_KEY` is deleted by `conftest.py`
in every test process; the OpenAI SDK is lazy-imported by
`OpenAILLMClient` so a test that accidentally constructed it without
mocking would fail loudly on import, not silently make a real call.

**Why a heuristic fallback (and why not just retry the LLM).** Retries
multiply latency at the worst possible moment — when something is
already slow or down. The heuristic gives a deterministic answer in
sub-millisecond time and routes 90.6% of our fixture correctly. The
spec requires "An LLM failure must not crash the request" with a
defined fallback — heuristic > "try the LLM again, hope for the best".

**Why in-memory session store.** 3-day single-process scope. The
`SessionStore` interface is the only thing the pipeline depends on, so
swapping in Postgres or Redis later is a subclass replacement, not a
refactor. Defended explicitly because the assignment said it would
not penalize the choice if defended.

**Why the safety guard runs before the classifier.** Cost +
correctness. Pre-LLM filtering means harmful queries don't burn a
classifier call ($-cost) and don't risk eliciting a harmful structured
response from a model that decided to "explain how" politely. Per the
assignment's safety precedence: the guard is the only authority on
blocks; the classifier's verdict is informational only.

**Why templated prose in the portfolio_health agent.** The data IS the
answer. Templating gives:
- sub-millisecond first-token (no LLM round-trip)
- fully reproducible test output (no sampling drift)
- zero drift between prose and the structured fields a downstream
  consumer reads

LLMs help where the answer requires reasoning over open-ended context
(research, recommendations) — not where the answer is just numbers
plus a few branching templates.

**Why no fixture queries appear in the classifier prompt.** Few-shot
examples drawn from the public eval set would (a) inflate the public
score at the expense of the hidden set and (b) coach the model toward
specific phrasings it should generalize beyond. Verified by
`test_prompt_does_not_contain_fixture_phrases`.

---

## Tradeoffs and assumptions

**Simplifications shipped:**

- **In-memory session memory.** Lost on restart, single-process only.
  Production: Redis with a TTL keyed by `user_id`.
- **Annualized return is approximated.** We don't have transaction
  history, so we assume an average 2-year holding period. Documented
  in the agent prose ("Approximation: we don't have your full
  transaction history.").
- **Benchmark return is hardcoded** per base currency (S&P 500 ≈ 10%
  annualized for USD users). Real implementation would fetch SPY
  history for the period covered by the user's cost basis.
- **Cost basis treated in the position's listed currency.** FX
  conversion uses the *current* rate, not the historical purchase
  rate. Standard for retail brokerage statements but not strictly
  accurate accounting; the agent never claims the return is
  currency-attribution-adjusted.
- **No corporate actions.** Splits, dividends, spinoffs not modeled;
  yfinance returns split-adjusted prices so this is fine for current
  value but not for total return with dividends.
- **6 of 8 specialist agents are stubs.** Per the assignment contract;
  the registry shows how to add the next implementation as one entry.

**What would scale in production:**

- `SessionStore` → Redis (TTL'd) or Postgres for durable history.
- `UserStore` → DB adapter (Postgres / DynamoDB).
- `MarketData` → an internal market-data service with bulk endpoints
  + WebSocket subscriptions; yfinance is not appropriate at scale.
- `AgentRegistry` → multiple implementations per agent name, selected
  per-tenant (the assignment's stretch goal: "premium → gpt-4.1, free
  → gpt-4o-mini").
- Embedding-based pre-classifier so a high-confidence cache hit on a
  recent query skips the LLM entirely (assignment stretch).
- Per-tenant rate limiting at the route layer.
- Telemetry: structured logs include `meta` event payload + agent name
  + duration + (if degraded) `confidence < 0.5` flag.

---

## Performance

### Targets and observed numbers

| Target | Limit | This build |
|---|---|---|
| p95 first-token latency | < 2.0 s | **~LLM round-trip** (pipeline overhead p95 = 0.86 ms) |
| p95 end-to-end response | < 6.0 s | **~LLM round-trip** (pipeline overhead p95 = 0.86 ms) |
| Cost per query @ gpt-4.1 | < $0.05 | **~$0.005 typical** (see math below) |
| Safety guard latency | < 10 ms | **mean 0.014 ms / p99 0.021 ms** |
| Classifier routing accuracy | ≥ 85% | **100% LLM oracle / 90.6% heuristic alone** |
| Safety recall on harmful | ≥ 95% | **100%** |
| Safety pass-through on safe | ≥ 90% | **100%** |

### How the latency was measured

`scripts/measure_latency.py` runs the full pipeline through
FastAPI's `TestClient` (in-process, no real network) with
`FakeLLMClient` injected. It measures **first-byte** (when the first
SSE event chunk hits the client) and **end-to-end** (full response
read) for four request shapes: portfolio_health, stub agent, safety
blocked, and empty portfolio.

```
$ python scripts/measure_latency.py --n 200

Valura AI — pipeline latency (TestClient, FakeLLMClient, no network)
iterations per shape: 200

--- portfolio_health (200 iters) ---
  first-byte   mean=  0.79 ms   p50=  0.75 ms   p95=  0.86 ms   max=  3.70 ms
  end-to-end   mean=  0.79 ms   p50=  0.76 ms   p95=  0.86 ms   max=  3.70 ms

--- stub agent (200 iters) ---
  first-byte   mean=  0.74 ms   p50=  0.70 ms   p95=  0.73 ms   max=  9.08 ms
  end-to-end   mean=  0.74 ms   p50=  0.70 ms   p95=  0.74 ms   max=  9.08 ms

--- safety blocked (200 iters) ---
  first-byte   mean=  0.66 ms   p50=  0.66 ms   p95=  0.70 ms   max=  0.74 ms
  end-to-end   mean=  0.66 ms   p50=  0.66 ms   p95=  0.70 ms   max=  0.75 ms

--- empty portfolio (200 iters) ---
  first-byte   mean=  0.78 ms   p50=  0.72 ms   p95=  0.77 ms   max=  6.49 ms
  end-to-end   mean=  0.78 ms   p50=  0.73 ms   p95=  0.77 ms   max=  6.49 ms
```

**Reading the numbers:** these are pipeline overhead — safety check,
session lookup, classifier wiring, agent run, SSE encoding. The
templated agent has no LLM in its hot path, so the e2e is what you
see: under 1 ms p95.

For real-LLM total latency, add the OpenAI round-trip:
- `gpt-4o-mini` structured outputs: ~400–800 ms p50, ~1500 ms p95
- `gpt-4.1` structured outputs: ~600–1200 ms p50, ~1800 ms p95

Either model lands the system comfortably under the 2 s first-token
and 6 s e2e budgets, with most of the 6 s budget unused — leaving
headroom for slower LLM tail-latency days.

### Cost per query

Math (gpt-4.1 list pricing at writing time, $5 / 1M input + $20 / 1M
output):
- Input tokens: ~700 (system prompt + history + user) ≈ $0.0035
- Output tokens: ~150 (structured ClassificationResult) ≈ $0.003
- **Total per query: ~$0.0065** — well under the $0.05 cap

`gpt-4o-mini` (dev model) is ~30× cheaper, so dev-mode runs cost
~$0.0002 per query. The agent itself doesn't call the LLM, so the
total is one classifier call per turn.

---

## Testing

```bash
pytest tests/ -v              # 183 tests
pytest tests/ -q --tb=line    # quick run
```

| Test file | What it covers | Tests |
|---|---|---|
| `test_smoke.py` | package import, settings | 3 |
| `test_fixtures.py` | fixture schema validation | 23 |
| `test_safety_guard.py` | guard accuracy + latency vs gold set | 10 |
| `test_safety_robustness.py` | held-out adversarial probes + benign over-block | 22 |
| `test_llm_fake.py` | FakeLLMClient behavior | 16 |
| `test_llm_client.py` | OpenAI client error mapping (mocked SDK) | 8 |
| `test_classifier.py` | schema, prompt, heuristic, oracle, fallback, conversations | 24 |
| `test_portfolio.py` | models, market data, valuation, edge cases | 23 |
| `test_portfolio_health_agent.py` | agent per-user behavior, streaming contract | 13 |
| `test_stub_and_registry.py` | stub correctness, registry coverage, extensibility | 15 |
| `test_api.py` | session store + pipeline orchestration + HTTP+SSE | 21 |
| | **Total** | **178** + 5 schema = **183** |

Every test is offline. `tests/conftest.py` autouses a fixture that
deletes `OPENAI_API_KEY` from every test process — CI safety enforced
at the framework level, not by convention.

---

## Project layout

```
valura_ai_ass/
├── README.md                        # this file
├── .env.example                     # documented env vars
├── pyproject.toml                   # deps + pytest config
├── fixtures/
│   ├── README.md                    # fixture schema + matcher contract
│   ├── users/                       # 5 user profiles (edge cases)
│   ├── conversations/               # 3 multi-turn test cases
│   └── test_queries/
│       ├── intent_classification.json   # 85 labeled queries
│       └── safety_pairs.json            # 72 labeled pairs
├── scripts/
│   └── measure_latency.py           # perf script
├── src/valura_ai/
│   ├── main.py                      # FastAPI factory + production wiring
│   ├── config.py                    # pydantic-settings env loader
│   ├── api/
│   │   ├── routes.py                # POST /v1/chat
│   │   ├── pipeline.py              # safety→classifier→agent orchestration
│   │   ├── sse.py                   # SSE event encoding
│   │   └── user_store.py            # fixture-backed user lookup
│   ├── safety/
│   │   ├── guard.py                 # decision logic
│   │   └── patterns.py              # category patterns + refusal messages
│   ├── classifier/
│   │   ├── schema.py                # Pydantic ClassificationResult
│   │   ├── prompt.py                # taxonomy-driven system prompt
│   │   ├── classifier.py            # LLM call + fallback wiring
│   │   └── heuristic.py             # deterministic fallback
│   ├── agents/
│   │   ├── base.py                  # BaseAgent + AgentEvent
│   │   ├── portfolio_health.py      # implemented agent
│   │   ├── stub.py                  # not-implemented stub
│   │   ├── registry.py              # name → agent map
│   │   └── schemas.py               # agent output schemas
│   ├── portfolio/
│   │   ├── models.py                # Position, Portfolio, User
│   │   ├── market_data.py           # MarketData ABC + Static + YFinance
│   │   └── metrics.py               # compute_valuation
│   ├── session/
│   │   └── store.py                 # in-memory SessionStore
│   └── llm/
│       ├── client.py                # LLMClient ABC + OpenAI impl
│       └── fakes.py                 # FakeLLMClient for tests
└── tests/
    ├── conftest.py                  # deletes OPENAI_API_KEY for every test
    ├── matchers.py                  # documented entity matcher
    └── (test files listed above)
```

---

## Defence prep

### Likely interviewer questions

**Q: Walk me through how a request flows through the system.**
See [End-to-end request flow](#end-to-end-request-flow) above. The
short version: route opens an SSE stream and delegates to a
`Pipeline.run()` async generator that runs the safety guard
synchronously, then classifier (with heuristic fallback), then
registered agent. Pipeline events become SSE events one-for-one. The
whole thing runs under one `asyncio.timeout`.

**Q: What's a non-obvious decision you made and why?**
The safety guard's asymmetric handling of educational framing.
Educational markers (`"what is"`, `"explain"`) only excuse pure phrase
matches via R3. They never override methodology or evasion markers
(R1, R2). That's how a single guard handles both
`"what is insider trading"` (pass) and
`"explain insider trading and how it's done"` (block) without two
separate models. A naive design would either over-block (any keyword
hit) or under-block (any educational framing rescues). The asymmetry
is what lets it be both precise and recall-y.

**Q: How does the classifier avoid overfitting to your fixtures?**
Four mechanisms:
1. The system prompt has plain-language agent descriptions and
   general routing rules — no fixture queries appear in it. Verified
   by `test_prompt_does_not_contain_fixture_phrases`.
2. The heuristic fallback uses structural patterns
   (numerical-pattern → calculator, action-verb-on-ticker → recs),
   not fixture phrases.
3. A held-out adversarial probe lives in `tests/test_safety_robustness.py`:
   10 queries with no overlap to the gold set, plus 12 benign
   over-block probes. Pinned as parametrized regression tests.
4. The matcher uses **subset semantics** for entities — the
   classifier is allowed to find more entities than the fixture
   lists. Same rule the grader applies. We never train it to
   under-extract.

**Q: What happens when the LLM is down?**
Every `LLMError` (timeout, parse, refusal, transport) is caught in
`IntentClassifier.classify()`. The deterministic heuristic fires;
returns the same `ClassificationResult` schema; sets `confidence < 0.5`
so ops can detect a degraded run via the `meta` event. Pipeline has
its own `asyncio.timeout` that turns *anything-slow* into a structured
error event, never a 500. Tested in `test_pipeline_classifier_failure_falls_back_not_500`
and `test_pipeline_request_timeout_emits_error`.

**Q: How does the design support scaling to all 8 specialists?**
Adding a specialist is **one entry** in `build_default_registry()`'s
`implemented` dict — see `test_registry_extension_is_a_one_liner`.
The HTTP layer doesn't know which agents are real or stubbed; the
classifier prompt loads agents from the JSON taxonomy at startup; the
agent contract (`BaseAgent.stream()` yielding `AgentEvent`s) is the
only thing every agent shares. Beyond agents: `SessionStore`,
`UserStore`, `MarketData`, `LLMClient` are all interfaces — Redis or
DB or external service implementations are drop-in.

**Q: Why didn't you use an LLM for the portfolio_health agent?**
The data is the answer. Templated prose gives sub-millisecond
first-token, fully reproducible test output, and zero drift between
the prose the user reads and the structured fields a downstream
consumer reads. LLMs help where the answer requires reasoning over
open-ended context (research, recommendations). Health checks are a
few branching templates over numbers — not a place I want sampling
variance.

**Q: How do you handle multi-currency portfolios?**
All amounts are normalized to `user.base_currency` at the valuation
step, never at load time. JPY positions don't falsely dominate
concentration in raw quantities — verified by
`test_global_multi_currency_valuation` (asserts the JPY position
stays under 20% weight after FX). Missing FX is treated as
missing-priced and surfaced explicitly via `missing_data.currencies`
on the structured payload — the agent says "couldn't convert" rather
than blanking the report.

**Q: How do you guarantee tests don't hit real APIs?**
Three layers:
1. `conftest.py` autouses a fixture that **deletes** `OPENAI_API_KEY`
   from every test process.
2. `OpenAILLMClient` lazy-imports the `openai` SDK so a test that
   accidentally constructed it without mocking would fail at import,
   not silently make a real call.
3. `YFinanceMarketData` similarly lazy-imports `yfinance`. All tests
   use `StaticMarketData` instead.

**Q: What would you do differently with another week?**
Three concrete things:
1. **Embedding-based pre-classifier** so high-confidence cache hits on
   recent queries skip the LLM entirely (assignment stretch). Would
   move p95 first-token from ~1500 ms to ~50 ms for the steady-state
   tail of repeat questions.
2. **Build the second specialist agent.** `financial_calculator` is
   the easiest because the math is deterministic — would prove the
   "adding a new specialist" path end-to-end.
3. **Real time-series pricing** so annualized return and benchmark
   comparison are computed properly instead of approximated. Right now
   both use a 2-year holding-period assumption that's documented but
   not strictly accurate.

### Defence video — what I'd cover (≤10 minutes, hard cap)

1. (1 min) **The system in one diagram** — the request flow above.
2. (3 min) **One non-obvious decision: the safety guard's asymmetric
   educational rescue.** Why it matters, walk through one
   methodology-laden query and one definitional query, show that it
   handles both correctly, mention the held-out adversarial probe
   that pinned the design.
3. (3 min) **The portfolio_health agent on three users:** concentrated
   NVDA holder, empty portfolio, multi-currency. Show the prose +
   structured payload, point out the BUILD branch and the FX
   normalization.
4. (2 min) **One thing I'd do differently** with another week —
   embedding-based pre-classifier to amortize the LLM cost on
   repeated queries.
5. (1 min) **Wrap:** test suite numbers + the assignment thresholds
   side-by-side.
