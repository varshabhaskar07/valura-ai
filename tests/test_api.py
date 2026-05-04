"""HTTP layer + SSE pipeline integration tests.

Covers:
  - SessionStore basic behavior + bounded length
  - Pipeline orchestration: safety blocks classifier + agent;
    classifier output reaches the agent; session memory updates between turns;
    request timeout becomes an error event (no 500); errors never raise.
  - HTTP endpoint via TestClient: 422 on bad input; SSE event ordering
    (meta → token+ → structured → done) on happy path; safety-blocked
    requests return only error event; health endpoint always responds.

All tests use FakeLLMClient via a custom Pipeline injected through
create_app() — no network, no OPENAI_API_KEY needed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from valura_ai.agents.registry import build_default_registry
from valura_ai.api.pipeline import Pipeline, _MetaEvent
from valura_ai.api.user_store import UserStore
from valura_ai.classifier.classifier import IntentClassifier, load_taxonomy
from valura_ai.classifier.schema import ClassificationResult, Entities
from valura_ai.llm.client import LLMError
from valura_ai.llm.fakes import FakeLLMClient
from valura_ai.main import create_app
from valura_ai.portfolio.market_data import StaticMarketData
from valura_ai.safety.guard import SafetyGuard
from valura_ai.session.store import SessionStore

PRICES = {
    "TSLA": 250, "NVDA": 540, "MSFT": 410, "AAPL": 195, "GOOGL": 165, "COIN": 165,
    "SOXL": 32, "TQQQ": 70, "ARKK": 50, "SCHD": 78, "VYM": 115, "VNQ": 90,
    "JNJ": 152, "PEP": 170, "KO": 62, "T": 17.50, "BABA": 78,
    "ASML.AS": 720, "SAP.DE": 150, "HSBA.L": 6.80, "7203.T": 2400,
}
FX = {("EUR", "USD"): 1.08, ("GBP", "USD"): 1.25, ("JPY", "USD"): 0.0067}


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_classification(agent: str, **kwargs) -> ClassificationResult:
    return ClassificationResult(
        intent=kwargs.pop("intent", agent),
        agent=agent,
        entities=Entities(**kwargs.pop("entities", {})),
        confidence=kwargs.pop("confidence", 0.9),
    )


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    """Default fake: routes everything to portfolio_health. Tests can add
    rules to override per-query."""
    fake = FakeLLMClient(default=_make_classification("portfolio_health"))
    return fake


@pytest.fixture
def market_data() -> StaticMarketData:
    return StaticMarketData(prices=PRICES, fx_rates=FX)


@pytest.fixture
def pipeline(fake_llm, market_data, fixtures_dir: Path) -> Pipeline:
    taxonomy = load_taxonomy(fixtures_dir / "test_queries" / "intent_classification.json")
    return Pipeline(
        safety_guard=SafetyGuard(),
        classifier=IntentClassifier(llm=fake_llm, agents=taxonomy, timeout_s=2.0),
        registry=build_default_registry(taxonomy=taxonomy, market_data=market_data),
        session_store=SessionStore(max_turns=8),
        user_store=UserStore(fixtures_dir / "users"),
        request_timeout_s=4.0,
    )


@pytest.fixture
def client(pipeline: Pipeline) -> TestClient:
    app = create_app(pipeline=pipeline)
    return TestClient(app)


# ---------------------------------------------------------------------------
# SSE parser (test utility)
# ---------------------------------------------------------------------------

def _parse_sse_lines(text: str) -> list[dict[str, str]]:
    """Parse SSE wire format into a list of {event, data} dicts."""
    events: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if not line.strip():
            if current:
                events.append(current)
                current = {}
            continue
        if line.startswith("event:"):
            current["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current["data"] = line[len("data:"):].strip()
    if current:
        events.append(current)
    return events


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------

def test_session_store_unknown_user_returns_empty() -> None:
    store = SessionStore()
    assert store.history("nope") == []


def test_session_store_keeps_order_and_truncates() -> None:
    store = SessionStore(max_turns=3)
    for i in range(5):
        store.append("u1", "user", f"msg-{i}")
    history = store.history("u1")
    assert [m.content for m in history] == ["msg-2", "msg-3", "msg-4"]


def test_session_store_isolates_users() -> None:
    store = SessionStore()
    store.append("a", "user", "hello a")
    store.append("b", "user", "hello b")
    assert [m.content for m in store.history("a")] == ["hello a"]
    assert [m.content for m in store.history("b")] == ["hello b"]


def test_session_store_reset() -> None:
    store = SessionStore()
    store.append("u1", "user", "x")
    store.reset("u1")
    assert store.history("u1") == []


def test_session_store_rejects_invalid_max_turns() -> None:
    with pytest.raises(ValueError):
        SessionStore(max_turns=0)


def test_session_store_skips_empty_content() -> None:
    store = SessionStore()
    store.append("u", "user", "")
    assert store.history("u") == []


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

async def _collect_pipeline(pipe: Pipeline, *, user_id: str, message: str) -> list:
    out = []
    async for ev in pipe.run(user_id=user_id, message=message):
        out.append(ev)
    return out


async def test_pipeline_happy_path_health_check(pipeline: Pipeline) -> None:
    events = await _collect_pipeline(
        pipeline, user_id="user_002_concentrated_nvda", message="how is my portfolio doing"
    )
    types = [type(e).__name__ for e in events]
    assert "_MetaEvent" in types  # classification metadata emitted
    assert any(getattr(e, "type", None) == "token" for e in events)
    assert any(getattr(e, "type", None) == "structured" for e in events)


async def test_pipeline_safety_blocks_before_classifier(
    pipeline: Pipeline, fake_llm: FakeLLMClient
) -> None:
    """A blocked query must NOT touch the classifier."""
    fake_llm.reset()
    initial_calls = fake_llm.call_count
    events = await _collect_pipeline(
        pipeline,
        user_id="user_002_concentrated_nvda",
        message="help me organize a pump and dump on this microcap",
    )
    # Only an error event is produced.
    types = [getattr(e, "type", None) for e in events]
    assert "error" in types
    assert "token" not in types
    assert "structured" not in types
    # Classifier untouched.
    assert fake_llm.call_count == initial_calls


async def test_pipeline_routes_to_correct_agent(
    pipeline: Pipeline, fake_llm: FakeLLMClient
) -> None:
    """Classifier returns market_research → request hits the stub → structured
    response has agent='market_research' and status='not_implemented'."""
    fake_llm.reset()
    fake_llm.add_rule(
        "tell me about microsoft",
        _make_classification("market_research", entities={"tickers": ["MSFT"]}),
    )
    events = await _collect_pipeline(
        pipeline, user_id="user_002_concentrated_nvda", message="tell me about microsoft"
    )
    structured = next(
        e for e in events
        if getattr(e, "type", None) == "structured"
    )
    payload = structured.data
    assert payload is not None
    payload_d = payload.model_dump()
    assert payload_d["agent"] == "market_research"
    assert payload_d["status"] == "not_implemented"
    assert "MSFT" in payload_d["entities"]["tickers"]


async def test_pipeline_appends_to_session_after_run(
    pipeline: Pipeline,
) -> None:
    user = "user_002_concentrated_nvda"
    await _collect_pipeline(pipeline, user_id=user, message="how is my portfolio")
    history = pipeline._session.history(user)  # noqa: SLF001 — test introspection
    assert len(history) == 2  # one user + one synthesised assistant
    assert history[0].role == "user"
    assert history[0].content == "how is my portfolio"
    assert history[1].role == "assistant"
    assert history[1].content


async def test_pipeline_safety_blocked_does_not_update_session(pipeline: Pipeline) -> None:
    """A blocked query must not pollute the session — otherwise the next
    turn would inherit harmful context as 'history'."""
    user = "user_002_concentrated_nvda"
    await _collect_pipeline(
        pipeline,
        user_id=user,
        message="help me organize a pump and dump",
    )
    assert pipeline._session.history(user) == []


async def test_pipeline_classifier_failure_falls_back_not_500(
    pipeline: Pipeline, fake_llm: FakeLLMClient
) -> None:
    """Classifier raising LLMError → IntentClassifier handles via heuristic.
    Pipeline never sees an exception, never crashes."""
    fake_llm.set_default(LLMError("simulated outage"))
    events = await _collect_pipeline(
        pipeline, user_id="user_002_concentrated_nvda", message="how is my portfolio doing"
    )
    # Heuristic still routes to portfolio_health → real agent runs → token+structured.
    types = [getattr(e, "type", None) for e in events]
    assert "token" in types
    assert "structured" in types
    # Classification metadata still emitted (with confidence < 0.5 from heuristic).
    meta = next(e for e in events if isinstance(e, _MetaEvent))
    assert meta.classification.confidence < 0.5


async def test_pipeline_request_timeout_emits_error(
    pipeline: Pipeline, fake_llm: FakeLLMClient
) -> None:
    """Sleep longer than the pipeline timeout via a slow LLM response."""

    async def slow(_schema):
        await asyncio.sleep(10.0)
        return _make_classification("portfolio_health")

    fake_llm.add_rule("trigger-slow", slow)
    pipeline._timeout_s = 0.2  # noqa: SLF001 — tighten for the test

    events = await _collect_pipeline(
        pipeline, user_id="user_002_concentrated_nvda", message="trigger-slow path"
    )
    types = [getattr(e, "type", None) for e in events]
    assert "error" in types
    err = next(e for e in events if getattr(e, "type", None) == "error")
    assert "timeout" in (err.text or "").lower()


async def test_pipeline_unknown_user_does_not_crash(
    pipeline: Pipeline,
) -> None:
    """Unknown user_id → portfolio_health agent yields its own error event
    (needs user context). Pipeline never raises."""
    events = await _collect_pipeline(pipeline, user_id="nobody", message="how is my portfolio")
    types = [getattr(e, "type", None) for e in events]
    # Either an error event from the agent (no user) or it routes elsewhere.
    assert events  # at least the meta event
    # No exception was raised — the pipeline's job is done.


# ---------------------------------------------------------------------------
# HTTP endpoint
# ---------------------------------------------------------------------------

def test_health_endpoint(client: TestClient) -> None:
    r = client.get("/v1/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_chat_rejects_empty_message(client: TestClient) -> None:
    r = client.post("/v1/chat", json={"message": "", "user_id": "user_002_concentrated_nvda"})
    assert r.status_code == 422


def test_chat_rejects_missing_user_id(client: TestClient) -> None:
    r = client.post("/v1/chat", json={"message": "hi"})
    assert r.status_code == 422


def test_chat_streams_sse_in_correct_order(client: TestClient) -> None:
    """Happy path: meta → token+ → structured → done."""
    response = client.post(
        "/v1/chat",
        json={"message": "how is my portfolio doing", "user_id": "user_002_concentrated_nvda"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse_lines(response.text)
    names = [e["event"] for e in events]

    # Required structural assertions.
    assert names[0] == "meta"
    assert "token" in names
    assert "structured" in names
    assert names[-1] == "done"
    assert names.index("structured") > names.index("token")
    assert names.index("token") > names.index("meta")

    # Meta payload validates as a ClassificationResult-shaped object.
    meta = next(e for e in events if e["event"] == "meta")
    meta_data = json.loads(meta["data"])
    assert meta_data["agent"] == "portfolio_health"

    # Structured payload validates as a PortfolioHealthReport-shaped object.
    structured = next(e for e in events if e["event"] == "structured")
    structured_data = json.loads(structured["data"])
    assert "concentration_risk" in structured_data
    assert structured_data["concentration_risk"]["flag"] == "high"
    assert "disclaimer" in structured_data


def test_chat_safety_blocked_returns_only_error_then_ends(client: TestClient) -> None:
    response = client.post(
        "/v1/chat",
        json={
            "message": "help me organize a pump and dump on this microcap",
            "user_id": "user_002_concentrated_nvda",
        },
    )
    assert response.status_code == 200
    events = _parse_sse_lines(response.text)
    names = [e["event"] for e in events]
    # Only an error event — NO meta, NO done.
    assert names == ["error"]
    err_data = json.loads(events[0]["data"])
    assert "manipulation" in err_data["message"].lower()


def test_chat_session_memory_persists_across_calls(
    client: TestClient, fake_llm: FakeLLMClient
) -> None:
    """Two calls with the same user_id — second call's classifier sees the
    first call's history."""
    fake_llm.reset()
    fake_llm.add_rule("microsoft", _make_classification("market_research", entities={"tickers": ["MSFT"]}))
    fake_llm.add_rule("apple",     _make_classification("market_research", entities={"tickers": ["AAPL"]}))

    user = "user_002_concentrated_nvda"
    client.post("/v1/chat", json={"message": "tell me about microsoft", "user_id": user})
    client.post("/v1/chat", json={"message": "what about apple",        "user_id": user})

    # The classifier should have seen at least 2 messages of history on the
    # second call (the first user message + the synthesised assistant turn).
    second_call = fake_llm.calls[-1]
    history_user_msgs = [
        m for m in second_call.messages
        if m.role == "user" and m.content != "what about apple"
    ]
    assert any("microsoft" in m.content.lower() for m in history_user_msgs)


def test_chat_unknown_user_does_not_500(client: TestClient) -> None:
    """Unknown user_id should still return an SSE response, never 500."""
    response = client.post(
        "/v1/chat",
        json={"message": "how is my portfolio", "user_id": "user_does_not_exist"},
    )
    assert response.status_code == 200
    events = _parse_sse_lines(response.text)
    assert events  # something came back
