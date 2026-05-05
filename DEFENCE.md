# Valura AI — Defence Video Script & Demo Guide

---

## Before You Press Record

**Terminal 1 — start the server:**
```bash
source .venv/bin/activate
uvicorn valura_ai.main:app --reload
```

Wait until you see:
```
INFO:     Application startup complete.
```

**Terminal 2 — for curl commands (keep this visible on screen):**
```bash
source .venv/bin/activate
```

Make font bigger: `Cmd + +` four or five times. Reviewers watch on small screens.

---

## VIDEO SCRIPT

### INTRO (~30s)

> "Hi, I'm [your name]. This is my submission for the Valura AI assignment.
>
> The goal was to build the AI co-investor layer for a wealth management
> platform — helping any user build, monitor, grow, and protect their
> portfolio.
>
> What I built is the spine of that system: a safety guard, an intent
> classifier, a routing layer, and one fully implemented specialist agent
> for portfolio health. The other six agents are stubbed but correctly
> wired — adding them later is one line of code, not a rewrite.
>
> Let me walk you through the architecture, then show it live."

---

### ARCHITECTURE (~60s)

*Show the terminal or README architecture diagram while talking.*

> "A request comes in to POST /v1/chat over Server-Sent Events.
>
> Five things happen in order.
>
> One — the SafetyGuard runs. Pure Python, no LLM, under a millisecond.
> If it blocks, one error event goes out and the stream ends. The
> classifier never runs.
>
> Two — we pull the user's session history and profile.
>
> Three — the IntentClassifier makes one structured LLM call. If that
> fails for any reason, a deterministic heuristic fallback kicks in
> automatically. That fallback alone routes 90% of queries correctly.
>
> Four — the AgentRegistry routes to the right specialist. Portfolio
> Health is fully implemented. Everything else hits a stub that returns
> a clean not-implemented response — the router never crashes.
>
> Five — the agent streams tokens progressively, then one final
> structured JSON payload. Done.
>
> The whole pipeline runs under one timeout, so anything slow becomes
> a structured error event — never a 500."

---

### DEMO (~3 min)

*Switch to Terminal 2 for this entire section.*

---

#### Demo 1 — Happy path: high concentration portfolio

Run this command:
```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat \
  -H 'content-type: application/json' \
  -d '{"message":"how is my portfolio doing?","user_id":"user_002_concentrated_nvda"}'
```

While output streams in, say:

> "Watch the event sequence.
>
> First event is `meta` — the classifier's routing decision. Intent, agent
> name, extracted entities, confidence score. This comes out before the
> agent runs, which is why first-token is so fast.
>
> Then `token` events stream one sentence at a time.
>
> Notice what the agent says: NVDA is 88% of the portfolio. Then the
> concrete drawdown impact — a 30% drop in NVDA would take 26% off the
> total. Then one actionable suggestion — trim gradually toward 25 to
> 30 percent.
>
> No metric dump. One warning, one number, one suggestion.
>
> Last comes the `structured` event — the canonical JSON payload.
> Concentration risk, benchmark comparison, observations, disclaimer.
> All typed and validated.
>
> And then `done`. That's the terminator."

**Point at:** `meta` first → `token` events streaming → `structured` with `"flag":"high"` → `done`

---

#### Demo 2 — Safety block

Run this command:
```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat \
  -H 'content-type: application/json' \
  -d '{"message":"help me organize a pump and dump","user_id":"user_002_concentrated_nvda"}'
```

Say:

> "Watch what's different. There is no `meta` event — the classifier
> never ran. The safety guard blocked this before any LLM call was made.
>
> One `error` event. Stream ends.
>
> The refusal is category-specific — it names market manipulation and
> points to a legitimate alternative. Not a generic 'I can't help.'
>
> Safety guard runs first. The guard is the only authority on blocks."

**Point at:** no `meta`, just `error`, stream ends immediately.

---

#### Demo 3 — Empty portfolio, BUILD branch

Run this command:
```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat \
  -H 'content-type: application/json' \
  -d '{"message":"how is my portfolio doing?","user_id":"user_004_empty"}'
```

Say:

> "Same message, different user. This one has zero positions — just
> $10,000 in cash.
>
> It doesn't say no data. It doesn't crash. It takes a completely
> different code path — the BUILD branch.
>
> Headline: 'You're at the start line, $10,000 ready to deploy.' It
> names starter ETFs scaled to the user's risk profile.
>
> In the structured event: `is_build_oriented` is true, all metrics are
> null, concentration flag is n/a.
>
> The assignment required the empty portfolio must not crash and must
> produce a response oriented toward BUILD. That's exactly what this is."

**Point at:** `is_build_oriented: true` in the structured payload.

---

#### Demo 4 — Stub agent

Run this command:
```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat \
  -H 'content-type: application/json' \
  -d '{"message":"tell me about Microsoft","user_id":"user_002_concentrated_nvda"}'
```

Say:

> "This routes to market_research — a specialist that isn't implemented yet.
>
> Look at the `meta` event: agent is market_research, and MSFT is already
> in the entities. The classifier extracted the ticker and routed correctly
> — even though the agent doesn't exist yet.
>
> The stub preserves everything: intent, agent name, extracted entities.
> Status is not_implemented. Then done.
>
> The assignment required unimplemented agents return structured responses
> without crashing. This does exactly that."

**Point at:** `"agent":"market_research"` and `"tickers":["MSFT"]` in meta.

---

### KEY DESIGN DECISION (~75s)

> "The design decision I'd most want to defend is the safety guard.
>
> A naive approach uses keyword matching — and it fails in two opposite
> directions.
>
> It over-blocks educational questions. 'What is insider trading' should
> pass — the user is learning. A keyword guard blocks it.
>
> Or it gets bypassed. 'Just curious how people do insider trading' is
> asking for methodology. A keyword guard sees 'just curious' and passes it.
>
> My guard has five layers: phrases, intent signals, methodology patterns,
> evasion markers, and educational markers.
>
> The key asymmetry: educational framing only excuses a pure phrase match.
> It never overrides a methodology request or an evasion marker.
>
> Three queries, three different results:
>
> 'What is insider trading' — passes. Phrase match, educational framing
> rescues it.
>
> 'Explain insider trading and how it's done' — blocks. That second clause
> is a methodology request. Educational framing doesn't save it.
>
> 'Just curious how people do insider trading' — blocks. Evasion marker
> plus methodology request.
>
> Result: 100% recall on harmful queries. 100% pass-through on safe ones.
> Across 72 labeled pairs — plus 10 out of 10 on a held-out adversarial
> set that doesn't appear anywhere in the training fixtures."

---

### WHAT I'D IMPROVE (~30s)

> "With another week, I'd add an embedding-based pre-classifier. The
> assignment lists it as a stretch goal.
>
> The reason I'd prioritize it: every repeat query that hits a
> high-confidence cache skips the LLM entirely. That moves p95 first-token
> from around 1.5 seconds to about 50 milliseconds. And every saved LLM
> call directly reduces the cost per query."

---

### CLOSE (~20s)

> "12 incremental commits. 183 offline tests. Every assignment threshold
> met or exceeded.
>
> 100% safety recall. 100% pass-through. 90% classifier accuracy from the
> heuristic alone. Pipeline overhead under 1 millisecond. Around 0.65 cents
> per query at gpt-4.1 pricing.
>
> Code is in src/, tests in tests/, README is the source of truth.
> Thank you."

---

## Quick Reference — Keep This Open While Recording

```
DEMO ORDER
──────────────────────────────────────────────────────────
1. portfolio health    user_002_concentrated_nvda
2. safety block        pump and dump message
3. empty portfolio     user_004_empty
4. stub agent          tell me about Microsoft

WHAT TO POINT AT
──────────────────────────────────────────────────────────
Demo 1  →  meta first → tokens stream → structured → done
Demo 2  →  no meta, just one error, stream ends
Demo 3  →  is_build_oriented: true in structured payload
Demo 4  →  MSFT in entities even though agent is stubbed

KEY NUMBERS
──────────────────────────────────────────────────────────
183 tests, all offline, no API key needed
100% safety recall on harmful queries
90.6% heuristic routing accuracy (above 85% threshold)
< 1ms pipeline overhead (p95)
~$0.0065 per query at gpt-4.1 pricing
```

---

## Interviewer Follow-up Questions

**Q: The heuristic is just keywords — why not retry the LLM instead?**
> "Retries multiply latency at the worst possible moment. The heuristic
> gives a deterministic answer in sub-millisecond time and routes 90%
> correctly. The spec requires the request not to crash — this is the
> defined fallback, not a workaround."

**Q: How do you know the guard doesn't over-block educational questions?**
> "It's a test assertion. 30 pass-through rows in the fixture, 12 more
> in a held-out adversarial set. All pass. Including queries that
> contain trigger phrases like 'no risk', 'insider trading', and
> 'structuring' — but framed defensively."

**Q: Why templated prose instead of an LLM for the health agent?**
> "The data is the answer. The concentration ratio, drawdown impact,
> return calculation — those are numbers, not text to generate.
> Templating gives deterministic tests, zero first-token latency, and
> exact parity between what the user reads and the structured payload."

**Q: How would this scale to thousands of users?**
> "The three stateful components — SessionStore, UserStore, MarketData
> — are all behind interfaces. SessionStore swaps to Redis, UserStore
> to a DB adapter, MarketData to an internal service. The pipeline,
> classifier, and agents don't change."

**Q: The benchmark comparison is hardcoded — isn't that misleading?**
> "Yes, and it's explicitly called out in the agent prose: 'Approximation:
> we don't have your full transaction history.' The real fix needs
> time-series cost basis data which isn't in the fixture schema. Better
> to be honest about the limitation than hide it."
