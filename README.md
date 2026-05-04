# Valura AI — Microservice (assignment build)

A FastAPI microservice that takes a financial query, runs it through a
safety guard, classifies intent with a single LLM call, routes to a
specialist agent, and streams the response back over Server-Sent Events.

This README is the single source of truth for setup, decisions, and
defence — per the assignment.

> **Status:** scaffolding in place. Modules land in subsequent commits;
> see `git log` for the build order.

---

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env  # OPENAI_API_KEY only needed for live runs, not tests
pytest tests/ -v
```

The full service (once Step 9 lands) runs with:

```bash
uvicorn valura_ai.main:app --reload
```

## Layout

```
src/valura_ai/        package
  api/                FastAPI routes + SSE encoder
  safety/             pure-Python safety guard
  classifier/         LLM intent classifier + heuristic fallback
  agents/             portfolio_health (implemented) + stub
  portfolio/          domain models, metrics, market data
  session/            in-memory conversation memory
  llm/                OpenAI client wrapper + offline fake
tests/                pytest, runs without OPENAI_API_KEY
fixtures/             user profiles, conversations, test queries
```

## Decisions (running list — expanded in later commits)

- **Python 3.11+ / FastAPI / sse-starlette** — async-first, native
  streaming, mature SSE handling.
- **In-memory session store for the demo** — defended in the relevant
  module; interface is swappable for Postgres/Redis without touching
  agent code.
- **Tests never touch the network.** A `FakeLLMClient` injected via
  dependency override; `OPENAI_API_KEY` is unset by `conftest.py`.

## Defence video

_Link added with the final commit._
