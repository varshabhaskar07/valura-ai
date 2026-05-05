# Defence — Video Script & Demo Guide

---

## Before You Record

**Terminal 1 — server:**
```bash
source .venv/bin/activate
uvicorn valura_ai.main:app --reload
```

**Terminal 2 — curl commands (keep visible on screen):**
```bash
source .venv/bin/activate
```

Increase font size: `Cmd + +` a few times.

---

## VIDEO SCRIPT (~7 min)

---

**INTRO (30s)**

> "I built Valura AI — the AI co-investor layer for a wealth management platform.
> The mission: help any user build, monitor, grow, and protect their portfolio.
>
> This build is the spine — safety guard, intent classifier, routing, and one
> fully implemented agent. The other six are stubbed behind the same interface,
> so adding them later is one line, not a rewrite."

---

**ARCHITECTURE (60s)**

> "A request hits POST /v1/chat over Server-Sent Events. Five steps:
>
> One — SafetyGuard runs. Pure Python, no LLM, under a millisecond.
> If it blocks, one error event goes out. Classifier never runs.
>
> Two — pull session history and user profile.
>
> Three — IntentClassifier makes one structured LLM call.
> If it fails, a deterministic heuristic kicks in. That fallback alone routes 90% correctly.
>
> Four — AgentRegistry routes to the right specialist.
> Portfolio Health is real. Everything else returns a clean not-implemented response.
>
> Five — agent streams tokens, then one final structured JSON payload.
>
> The whole pipeline runs under one timeout. Anything slow becomes a structured
> error event — never a 500."

---

**KEY DESIGN DECISION (75s)**

> "The decision I'd most want to defend is the safety guard.
>
> A keyword guard fails two ways — it over-blocks educational questions,
> or gets bypassed by methodology requests dressed as educational.
>
> My guard has five layers: phrases, intent, methodology, evasion, educational.
> The key asymmetry: educational framing only excuses a pure phrase match.
> It never overrides a methodology request or evasion marker.
>
> Three queries, three results:
>
> 'What is insider trading' — passes. Phrase match, educational framing rescues it.
>
> 'Explain insider trading and how it's done' — blocks.
> That second clause is a methodology request. Educational framing doesn't save it.
>
> 'Just curious how people do insider trading' — blocks.
> Evasion marker plus methodology request.
>
> Result: 100% recall on harmful. 100% pass-through on safe.
> Across 72 labeled pairs, plus 10 out of 10 on a held-out adversarial set."

---

**DEMO (90s)**

*[Run the curl commands — narrate while output streams]*

> "Watch the event sequence — meta first, then tokens one sentence at a time,
> then the structured payload, then done.
>
> The concentrated user: NVDA is 88% of the book. The agent says the drawdown
> impact in dollars, then one actionable suggestion. No metric dump.
>
> The empty portfolio: completely different code path — the BUILD branch.
> 'You're at the start line, $10,000 ready to deploy.' Not a degraded response.
>
> The safety block: no meta, no tokens, just one error event. Stream ends.
> Category-specific refusal, not a generic message.
>
> The stub agent: routes to market_research correctly, MSFT extracted in entities,
> status is not_implemented. Router never crashes."

---

**IMPROVEMENT (30s)**

> "With another week I'd add an embedding-based pre-classifier.
> High-confidence cache hits skip the LLM entirely.
> That moves p95 first-token from 1.5 seconds to 50 milliseconds,
> and every saved call directly hits the cost target."

---

**CLOSE (20s)**

> "12 commits. 183 offline tests. Every threshold exceeded.
> 100% safety recall. 90% heuristic accuracy. Under a millisecond pipeline overhead.
> 0.65 cents per query at gpt-4.1. Thank you."

---

## DEMO COMMANDS

Run these in order during the demo section:

**1. Portfolio health:**
```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat \
  -H 'content-type: application/json' \
  -d '{"message":"how is my portfolio doing?","user_id":"user_002_concentrated_nvda"}'
```

**2. Safety block:**
```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat \
  -H 'content-type: application/json' \
  -d '{"message":"help me organize a pump and dump","user_id":"user_002_concentrated_nvda"}'
```

**3. Empty portfolio:**
```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat \
  -H 'content-type: application/json' \
  -d '{"message":"how is my portfolio doing?","user_id":"user_004_empty"}'
```

**4. Stub agent:**
```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat \
  -H 'content-type: application/json' \
  -d '{"message":"tell me about Microsoft","user_id":"user_002_concentrated_nvda"}'
```

---

## WHAT TO POINT AT

| Demo | Point at |
|---|---|
| Portfolio health | `meta` first → tokens streaming → `"flag":"high"` → `done` |
| Safety block | No `meta`, just `error`, stream ends immediately |
| Empty portfolio | `is_build_oriented: true` in structured payload |
| Stub agent | `"agent":"market_research"` and `"tickers":["MSFT"]` in meta |

---

## FOLLOW-UP Q&A

**Q: Why heuristic fallback instead of retrying the LLM?**
> Retries multiply latency when things are already broken. The heuristic answers in sub-millisecond time and routes 90% correctly. The spec requires no crash — this is the defined fallback.

**Q: How do you know the guard doesn't over-block educational questions?**
> It's a test assertion. 30 pass-through rows in the fixture, 12 more in a held-out set. All pass — including queries containing trigger phrases like 'no risk' and 'insider trading' framed defensively.

**Q: Why templated prose instead of an LLM for the health agent?**
> The data is the answer. Templating gives deterministic tests, zero first-token latency, and exact parity between what the user reads and the structured payload.

**Q: How does this scale to all 8 specialists?**
> Adding a specialist is one entry in the registry's implemented dict. The HTTP layer, classifier, and pipeline don't change. SessionStore swaps to Redis, UserStore to a DB adapter — both are interfaces.

**Q: The benchmark return is hardcoded — isn't that misleading?**
> Yes, and it's called out explicitly in the agent prose: 'Approximation: we don't have your full transaction history.' Better to be honest than hide it.
