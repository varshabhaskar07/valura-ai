# Valura AI — Microservice

A FastAPI microservice that turns a financial query into a streamed,
grounded response. Pipeline: synchronous safety guard → single-call
intent classifier (with deterministic fallback) → routed specialist
agent → SSE stream.

This build is the **spine** of the assignment: safety + classifier +
routing + one fully-implemented agent (Portfolio Health) + the HTTP
layer. Adding the other six specialists later is one line of
registration, not a rewrite.

> **Defence video:** https://drive.google.com/file/d/1UQLrxrpfY2iG-ioOQam5xFSjRhaarQEn/view?usp=sharing

---

## Contents

- [Quickstart](#quickstart)
- [Demo](#demo)
- [Architecture](#architecture)
- [Request flow](#request-flow)
- [Components](#components)
- [Library choices](#library-choices)
- [Design decisions](#design-decisions)
- [Tradeoffs](#tradeoffs)
- [Performance](#performance)
- [Testing](#testing)
- [Project layout](#project-layout)
- [Submission checklist](#submission-checklist)

---

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env     # OPENAI_API_KEY only needed for live LLM; tests don't need it
pytest tests/ -v         # 183 tests, fully offline
uvicorn valura_ai.main:app --reload
```

Without `OPENAI_API_KEY` the service still runs — it falls back to a
stub LLM that routes everything to `portfolio_health`. Useful for
demoing the streaming pipeline without a key.

---

## Demo

### Request

```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat \
  -H 'content-type: application/json' \
  -d '{"message":"how is my portfolio doing?","user_id":"user_002_concentrated_nvda"}'
```

### Response (SSE, abridged)

```
event: meta
data: {"intent":"health_check","agent":"portfolio_health","entities":{...},
       "confidence":0.9,"safety_verdict":{"is_safe":true}}

event: token
data: Your portfolio is worth $216,575, +121.7% since cost basis — concentration is high.

event: token
data: NVDA is 88% of your portfolio — that single position carries most of your outcome.
       A 30% drop in NVDA would take roughly 26% off your total at a 30% drawdown.

event: token
data: Approximate annualized return is +60.8% — ahead of the S&P 500 by 50.8%
       on a similar-period basis. (Approximation: we don't have your full transaction history.)

event: token
data: Worth thinking about gradually trimming the top position to bring it closer
       to a 25–30% weight, depending on your goals.

event: token
data: This is not investment advice. Information is provided for educational purposes
       only; consult a licensed financial advisor before making any investment decisions.

event: structured
data: {"user_id":"user_002_concentrated_nvda",
       "headline":"Your portfolio is worth $216,575, +121.7% since cost basis...",
       "concentration_risk":{"top_position_pct":87.75,"top_3_positions_pct":98.08,"flag":"high"},
       "performance":{"total_return_pct":121.69,"annualized_return_pct":60.84},
       "benchmark_comparison":{"benchmark":"S&P 500","alpha_pct":50.84},
       "observations":[...],"is_build_oriented":false,"disclaimer":"..."}

event: done
data:
```

### Event flow

| Event | When | Payload |
|---|---|---|
| `meta` | After classifier resolves | `ClassificationResult` (routing decision + extracted entities) |
| `token` | Repeated, as the agent streams prose | One sentence per chunk |
| `structured` | Once at the end | The agent's canonical structured payload |
| `done` | Last, on success | empty body — terminator |
| `error` | On block / failure / timeout | `{"message": "..."}`; stream ends, no `done` |

Three other request shapes the live demo produces:

- **Stub agent** (`tell me about microsoft`) → `meta`(agent=`market_research`) → 1 `token` (stub message + entities echoed) → `structured` (`status: not_implemented`, MSFT preserved) → `done`
- **Safety blocked** (`help me organize a pump and dump`) → single `error` event, stream ends — no `meta`, no `done`
- **Empty portfolio** (`user_004`) → `meta` → `token+` → `structured` (`is_build_oriented: true`, all metrics `null`, observations name VTI/VXUS/BND) → `done`

---

## Architecture

```
                     POST /v1/chat  { message, user_id }
                                 │
                                 ▼
                       ┌─────────────────────┐
                       │      Pipeline       │  asyncio.timeout(12s)
                       └──────────┬──────────┘
                                  │
   ┌──────────────────────────────┼──────────────────────────────────┐
   │                              ▼                                  │
   │  1. SafetyGuard.check(message)                                  │
   │       ├─ blocked → emit `error` → END                           │
   │       └─ pass → continue                                        │
   │                                                                 │
   │  2. SessionStore.history(user_id)        — last N turns         │
   │  3. UserStore.get(user_id)               — profile + holdings   │
   │                              ▼                                  │
   │  4. IntentClassifier                                            │
   │       ├─ LLM (timeout 4s, structured output)                    │
   │       │     ├─ success → ClassificationResult                   │
   │       │     └─ LLMError → heuristic (deterministic, 90.6%)      │
   │       └─ unknown agent in response → heuristic                  │
   │                              ▼ emit `meta`                      │
   │                                                                 │
   │  5. AgentRegistry.get(classification.agent)                     │
   │       ├─ portfolio_health → real PortfolioHealthAgent           │
   │       └─ everything else  → StubAgent("not_implemented")        │
   │                              ▼ stream                           │
   │     Agent.stream() yields token+ then structured                │
   │                              ▼ emit per event                   │
   │                                                                 │
   │  6. SessionStore.append(user, assistant)                        │
   │                              ▼                                  │
   │                       emit `done`                               │
   └─────────────────────────────────────────────────────────────────┘
```

---

## Request flow

The route in `api/routes.py` is intentionally thin — parse body, open
SSE response, delegate to `Pipeline.run()`. The pipeline is one async
generator wrapped in `asyncio.timeout(request_timeout_s)` so a slow
component anywhere becomes a structured `error` event, never a 500.

1. **SafetyGuard.check()** — pure-Python, sub-millisecond. Blocked
   → one `error` event, stream ends. Classifier never runs (assignment
   §Safety Precedence). Session is **not** updated on a block.
2. **SessionStore.history()** — last N turns for this user.
3. **UserStore.get()** — profile + portfolio. `None` for unknown user
   is fine — pipeline still proceeds.
4. **IntentClassifier.classify()** — one LLM call, structured output.
   Any `LLMError` triggers the heuristic. Out-of-vocab agent in the
   response also triggers the heuristic. Pipeline emits a `meta`
   event with the classification.
5. **AgentRegistry.get()** — always returns a `BaseAgent`.
   `portfolio_health` is the real implementation; the other 7 are
   stubs. Agent's `stream()` yields `token+` then one `structured`
   event; the route translates each into the matching SSE event.
6. **SessionStore.append()** — record user message + a synthesised
   assistant turn (capped 500 chars) so the next call sees real
   conversation context.

Route emits `event: done` on success. On any `error` event, the route
exits its generator immediately — no trailing `done`.

---

## Components

### SafetyGuard — `src/valura_ai/safety/`

Pure-Python, no LLM, no I/O. Layered design:

| Layer | What it matches | Effect |
|---|---|---|
| **1. Phrases** | "insider trading", "pump and dump", "launder cash" | weight 1 |
| **2. Intent** | actor + action — "CFO friend hinted", "I have inside info" | weight 2 |
| **3. Methodology** | "how do I", "how would someone", "common methods" | rule trigger |
| **4. Evasion** | "just curious", "theoretically", "between you and me" | rule trigger |
| **5. Educational** | "what is", "explain the difference", "regulatory", named cases | rescue trigger |

Decision precedence (first match wins):
- **R1)** phrase ≥ 1 AND methodology → BLOCK *(catches "explain X and how it's done")*
- **R2)** phrase ≥ 1 AND evasion → BLOCK *(catches "just curious how X works")*
- **R3)** score ≥ 2 AND not educational → BLOCK
- **R4)** otherwise → PASS

**The asymmetry is the key idea:** educational framing only excuses
pure phrase matches, never methodology requests or evasion phrasing.
That's how a single guard handles both `"what is insider trading"`
(pass) and `"explain insider trading and how it's done"` (block).

7 categories, each with a distinct, professional refusal that points
to a legitimate alternative.

**Numbers** (against `fixtures/test_queries/safety_pairs.json`, 72 pairs):
- Recall on harmful: **100%** (42/42) — threshold ≥95%
- Pass-through on safe: **100%** (30/30) — threshold ≥90%
- Latency: mean **0.014 ms** / p99 **0.021 ms** — budget <10 ms
- 10/10 on a held-out adversarial set + 12/12 on benign over-block probes

### IntentClassifier — `src/valura_ai/classifier/`

One LLM call per request, Pydantic-validated structured output. The
prompt is **taxonomy-driven** — agent names load from
`fixtures/test_queries/intent_classification.json` at startup. Plain
descriptions of each agent's responsibility plus general routing
principles. **No fixture queries appear in the prompt** — pinned by
`test_prompt_does_not_contain_fixture_phrases`.

`ClassificationResult`:

```python
intent: str                    # short slug describing the user's ask
agent: str                     # one of the taxonomy agents (exact match)
entities:
    tickers: list[str]
    topics: list[str]
    sectors: list[str]
    amount: float | None
    rate: float | None
    period_years: float | None
confidence: float (0..1)
reasoning: str                 # short, internal-use
safety_verdict: { is_safe, note }   # informational only
```

**Heuristic fallback** triggers on any `LLMError` and on out-of-vocab
agent names. Pure Python, deterministic, sub-millisecond. Falls back
to the last user turn in history when the current turn is vague —
handles `"thoughts?"`, `"and now?"`, `"explain like im 5"`. Heuristic
confidence is capped at 0.4 so ops can detect a degraded run via the
`meta` event.

**Numbers** (85-query fixture):
- LLM oracle (mechanical wiring): 100% routing, 0 entity-match failures
- Heuristic alone: **77/85 = 90.6%** — already above the 85% threshold

### Portfolio domain — `src/valura_ai/portfolio/`

Pydantic models (`Position` frozen, `Portfolio` mutable). One
`MarketData` interface, two implementations: `StaticMarketData`
(in-memory, used by every test) and `YFinanceMarketData` (lazy-imports
yfinance, disk-cached 1h prices / 6h FX). `compute_valuation()` is the
single entry point — `User` + `MarketData` → typed `PortfolioValuation`.

Edge cases handled by typed-result fields, not exceptions:

| Edge case | Behaviour |
|---|---|
| Empty portfolio | `concentration.flag = "n/a"`, all metrics `None`, no crash |
| Single position | top1 = top3 = 100%, no `/0` |
| Missing price | in `missing_prices`, aggregates over priced subset |
| Missing FX | in `missing_fx`, position treated as missing-priced |
| Multi-currency | normalised to `user.base_currency` before weighting |

### Agents — `src/valura_ai/agents/`

`BaseAgent.stream()` is an async generator yielding `AgentEvent`s
(token / structured / error / info). The HTTP layer maps these into
SSE events.

**`PortfolioHealthAgent`** — implemented, deterministic, no LLM in its
hot path. Templated prose because the data IS the answer. Branches:

- **Empty** → BUILD branch: forward-looking headline, starter ETFs scaled to risk profile
- **High concentration** → top position by name + drawdown impact + trim suggestion
- **Moderate / low** → balanced framing or affirmation with one nuance
- **Aggressive risk + leveraged ETFs** → leverage note (neutral, not lecturing)
- **Cash > 25%** → cash-drag observation
- **Missing data** → warning observation, never a crash

Observations capped at 4, sorted critical > warning > info. Disclaimer
appended verbatim to every response.

**`StubAgent`** — stands in for every taxonomy agent that isn't
implemented. Returns the structured "not implemented" payload the
assignment specifies (intent, agent, entities, status, message) plus
prose that names the agent and echoes captured entities so the user
sees their request was understood.

`AgentRegistry` maps every taxonomy name to a `BaseAgent`. Adding a
new specialist is one entry in `build_default_registry()`'s
`implemented` dict — no other code changes.

### HTTP + SSE pipeline — `src/valura_ai/api/`, `src/valura_ai/main.py`

`Pipeline.run()` orchestrates everything in one async generator. The
route function is thin: parse, open SSE, forward. `api/sse.py` handles
event encoding — every Pydantic payload goes out via
`model_dump_json` so the wire format is stable. See [Demo](#demo) for
the wire-format examples.

---

## Library choices

| Library | Why |
|---|---|
| **FastAPI** | Async-native, OpenAPI for free, the standard ASGI stack — every reviewer can run it. |
| **uvicorn[standard]** | Production ASGI server; `[standard]` adds httptools + uvloop. |
| **sse-starlette** | Battle-tested SSE: content-type, keep-alives, disconnect handling. Worth the dep over rolling our own. |
| **openai** (≥1.50) | Structured outputs (`beta.chat.completions.parse`) returns Pydantic-validated objects in one round-trip — no JSON-from-prose parsing. |
| **pydantic v2** | Schemas at every boundary. Validation at the edge means agent code can trust its inputs. |
| **pydantic-settings** | Env-driven `Settings` with type validation — no config glue. |
| **yfinance** | Free, no key, fine for a 3-day build. Disk-cached so repeats don't re-fetch. |
| **httpx** + FastAPI's `TestClient` | Streaming-capable test client — integration tests parse real SSE wire format. |
| **pytest** + `pytest-asyncio` | Async tests without ceremony. `asyncio_mode = "auto"`. |

---

## Design decisions

**FastAPI over alternatives.** Native async, native streaming via ASGI,
type-driven request validation via Pydantic. Removes the boilerplate
between HTTP body and classifier/agent code.

**SSE over WebSockets.** SSE is one-way streaming over plain HTTP —
half the complexity of WebSockets for the same delivery guarantee.
`curl -N` and every browser consume it without a library. WebSockets
buy nothing here.

**`FakeLLMClient` for tests.** Same `LLMClient` interface as the real
client. Programmable per-test (substring / regex / callable matchers,
BaseModel / Exception / async-factory responses), records every call
for introspection. `OPENAI_API_KEY` is deleted by `conftest.py` in
every test process; OpenAI SDK is lazy-imported so a test that
accidentally constructed it without mocking would fail loudly on
import, not silently make a real call.

**Heuristic fallback over retries.** Retries multiply latency at the
worst possible moment. The heuristic gives a deterministic answer in
sub-millisecond time and routes 90.6% of our fixture correctly.
Assignment requires "An LLM failure must not crash the request" — this
is the defined fallback.

**In-memory session store.** 3-day single-process scope. The
`SessionStore` interface is the only thing the pipeline depends on, so
swapping in Redis or Postgres later is an interface implementation,
not a refactor. The assignment said it would not penalize this if
defended.

**Safety guard before classifier.** Cost + correctness. Pre-LLM
filtering means harmful queries don't burn a classifier call and don't
risk eliciting a harmful structured response. Per the assignment's
safety precedence: the guard is the only authority on blocks.

**Templated prose in `portfolio_health`.** The data IS the answer.
Templating gives sub-millisecond first-token, fully reproducible test
output, and zero drift between prose and the structured fields a
downstream consumer reads. LLMs help where the answer needs reasoning
over open-ended context (research, recommendations) — not where it's
numbers and a few branching templates.

**No fixture queries in the classifier prompt.** Few-shot examples
from the public eval set would inflate the public score at the
expense of the hidden set. Pinned by
`test_prompt_does_not_contain_fixture_phrases`.

---

## Tradeoffs

**Simplifications shipped:**

- **In-memory session memory** — lost on restart, single-process. Production: Redis with a TTL keyed by `user_id`.
- **Annualized return is approximated** — we don't have transaction history; assume ~2y average holding. Documented in the agent prose.
- **Benchmark return is hardcoded per base currency** — S&P 500 ≈ 10% annualized for USD. Real impl would fetch SPY history for the period.
- **Cost basis treated in the position's listed currency** — FX uses *current* rate, not historical purchase rate. Standard for retail brokerage statements.
- **No corporate actions** — splits, dividends, spinoffs not modeled. yfinance prices are split-adjusted so current value is fine; total return excludes dividends.
- **6 of 8 agents are stubs** — per the assignment contract; the registry shows how to add the next implementation as one entry.

**What would scale in production:**

- `SessionStore` → Redis (TTL'd) or Postgres for durable history
- `UserStore` → DB adapter (Postgres / DynamoDB)
- `MarketData` → internal market-data service with bulk + WebSocket subscriptions; yfinance is not appropriate at scale
- `AgentRegistry` → multiple implementations per agent name, selected per-tenant (premium → `gpt-4.1`, free → `gpt-4o-mini`)
- Embedding-based pre-classifier so high-confidence cache hits skip the LLM entirely
- Per-tenant rate limiting at the route layer
- Structured logs include `meta` event payload + agent name + duration + degraded-run flag

---

## Performance

### Targets vs observed

| Target | Limit | This build |
|---|---|---|
| p95 first-token latency | < 2.0 s | **~LLM round-trip** (pipeline overhead p95 = 0.86 ms) |
| p95 end-to-end | < 6.0 s | **~LLM round-trip** (pipeline overhead p95 = 0.86 ms) |
| Cost per query @ gpt-4.1 | < $0.05 | **~$0.0065** (math below) |
| Safety guard latency | < 10 ms | **mean 0.014 ms / p99 0.021 ms** |
| Classifier routing accuracy | ≥ 85% | **100% LLM oracle / 90.6% heuristic alone** |
| Safety recall on harmful | ≥ 95% | **100%** (42/42) |
| Safety pass-through on safe | ≥ 90% | **100%** (30/30) |

### How the latency was measured

`scripts/measure_latency.py` runs the full pipeline through FastAPI's
`TestClient` (in-process, no real network) with `FakeLLMClient`
injected. It measures **first-byte** (when the first SSE chunk hits
the client) and **end-to-end** (full response read) for four request
shapes.

```
$ python scripts/measure_latency.py --n 200
──────────────────────────────────────────────────────────────────────────────
  Valura AI — /v1/chat latency  (TestClient · FakeLLMClient · no network)
  iterations per shape: 200
──────────────────────────────────────────────────────────────────────────────
  shape                 metric              mean      p50      p95      p99      max
  --------------------------------------------------------------------------
  portfolio_health      first-byte       0.85 ms  0.80 ms  1.04 ms  3.63 ms  4.40 ms
  portfolio_health      end-to-end       0.86 ms  0.80 ms  1.05 ms  3.64 ms  4.40 ms
  stub agent            first-byte       0.75 ms  0.70 ms  0.76 ms  0.78 ms  9.64 ms
  stub agent            end-to-end       0.75 ms  0.70 ms  0.76 ms  0.78 ms  9.65 ms
  safety blocked        first-byte       0.66 ms  0.66 ms  0.69 ms  0.77 ms  0.83 ms
  safety blocked        end-to-end       0.66 ms  0.66 ms  0.70 ms  0.78 ms  0.83 ms
  empty portfolio       first-byte       0.78 ms  0.72 ms  0.77 ms  4.47 ms  7.37 ms
  empty portfolio       end-to-end       0.78 ms  0.73 ms  0.77 ms  4.48 ms  7.37 ms
  --------------------------------------------------------------------------
  ALL SHAPES            first-byte       0.76 ms  0.71 ms  0.88 ms  1.25 ms  9.64 ms
  ALL SHAPES            end-to-end       0.76 ms  0.72 ms  0.88 ms  1.25 ms  9.65 ms
──────────────────────────────────────────────────────────────────────────────
```

These are pipeline overhead — safety + session lookup + classifier
wiring + agent + SSE encoding. The templated agent has no LLM in its
hot path, so e2e is what you see: under 1 ms p95. For real-LLM total
latency, add the OpenAI round-trip:

- `gpt-4o-mini` structured outputs: ~400–800 ms p50, ~1500 ms p95
- `gpt-4.1` structured outputs: ~600–1200 ms p50, ~1800 ms p95

Either model lands the system well under the 2 s first-token and 6 s
e2e budgets, with most of the 6 s budget unused for slow-LLM tail
days.

### Cost per query

Math at gpt-4.1 list pricing ($5 / 1M input + $20 / 1M output):
- Input ~700 tokens (system prompt + history + user) ≈ $0.0035
- Output ~150 tokens (`ClassificationResult`) ≈ $0.003
- **Total: ~$0.0065 per query** — well under the $0.05 cap

`gpt-4o-mini` (dev model) is ~30× cheaper. The portfolio_health agent
adds zero LLM calls.

---

## Testing

```bash
pytest tests/ -v              # 183 tests
pytest tests/ -q --tb=line    # quick run
```

| Test file | Covers | Tests |
|---|---|---:|
| `test_smoke.py` | package import, settings | 3 |
| `test_fixtures.py` | fixture schema validation | 23 |
| `test_safety_guard.py` | guard accuracy + latency vs gold set | 10 |
| `test_safety_robustness.py` | held-out adversarial + benign over-block probes | 22 |
| `test_llm_fake.py` | FakeLLMClient behaviour | 16 |
| `test_llm_client.py` | OpenAI client error mapping (mocked SDK) | 8 |
| `test_classifier.py` | schema, prompt, heuristic, oracle, fallback, conversations | 24 |
| `test_portfolio.py` | models, market data, valuation, edge cases | 23 |
| `test_portfolio_health_agent.py` | per-user behaviour, streaming contract | 13 |
| `test_stub_and_registry.py` | stub correctness, registry coverage, extensibility | 15 |
| `test_api.py` | session store + pipeline + HTTP+SSE | 21 |
| | **Total** | **183** |

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
│   └── test_queries/                # 85 classification + 72 safety pairs
├── scripts/
│   └── measure_latency.py           # perf script
├── src/valura_ai/
│   ├── main.py                      # FastAPI factory + production wiring
│   ├── config.py                    # pydantic-settings env loader
│   ├── api/                         # routes, pipeline, SSE encoding, user store
│   ├── safety/                      # SafetyGuard + patterns
│   ├── classifier/                  # schema, prompt, classifier, heuristic
│   ├── agents/                      # base + portfolio_health + stub + registry
│   ├── portfolio/                   # models, market_data, metrics
│   ├── session/                     # in-memory store
│   └── llm/                         # client + OpenAI impl + fake
└── tests/                           # 183 offline tests + matcher utility
```

---

## Submission checklist

- [ ] `pip install -e ".[dev]"` succeeds on a clean Python 3.11+ venv
- [ ] `pytest tests/ -v` passes — **183 tests, 0 failures**, no `OPENAI_API_KEY` needed
- [ ] `uvicorn valura_ai.main:app --reload` starts without errors
- [ ] `curl -N -X POST http://127.0.0.1:8000/v1/chat -H 'content-type: application/json' -d '{"message":"how is my portfolio doing?","user_id":"user_002_concentrated_nvda"}'` returns a valid SSE stream (`meta` → `token+` → `structured` → `done`)
- [ ] Safety-blocked query (`pump and dump` phrasing) returns a single `error` event
- [ ] `python scripts/measure_latency.py` reports p95 first-byte under 2 ms (pipeline overhead)
- [ ] Repo is clean: no committed `.env`, no `.venv/`, no `__pycache__/`, no `.cache/`
- [ ] `git log --oneline` shows 12 incremental commits — not one final dump
- [ ] **Defence video link added** to the top of this README and tested
- [ ] Repo pushed to remote, accessible to reviewers
