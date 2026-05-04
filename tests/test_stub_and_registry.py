"""StubAgent + AgentRegistry tests.

The router's job is to route correctly even when the destination is a
stub. These tests assert that:
  - every taxonomy agent resolves through the registry
  - the stub never crashes, regardless of input shape
  - the structured payload preserves classifier output (intent, agent,
    entities) so downstream consumers don't lose information
  - unknown agent names get a safe stub fallback
  - extending the registry is a one-liner (build_default_registry shown)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from valura_ai.agents.base import AgentEvent, BaseAgent
from valura_ai.agents.portfolio_health import PortfolioHealthAgent
from valura_ai.agents.registry import AgentRegistry, build_default_registry
from valura_ai.agents.schemas import StubResponse
from valura_ai.agents.stub import StubAgent
from valura_ai.classifier.classifier import load_taxonomy
from valura_ai.classifier.schema import ClassificationResult, Entities
from valura_ai.portfolio.market_data import StaticMarketData


@pytest.fixture(scope="module")
def taxonomy(fixtures_dir: Path) -> list[str]:
    return load_taxonomy(fixtures_dir / "test_queries" / "intent_classification.json")


@pytest.fixture(scope="module")
def market_data() -> StaticMarketData:
    return StaticMarketData(prices={}, fx_rates={})


def _classification(agent: str, **kwargs) -> ClassificationResult:
    return ClassificationResult(
        intent=kwargs.pop("intent", agent),
        agent=agent,
        entities=Entities(**kwargs.pop("entities", {})),
        **kwargs,
    )


async def _collect(agent: BaseAgent, classification: ClassificationResult) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    async for ev in agent.stream(classification=classification):
        events.append(ev)
    return events


# ---------------------------------------------------------------------------
# StubAgent
# ---------------------------------------------------------------------------

def test_stub_requires_agent_name() -> None:
    with pytest.raises(ValueError):
        StubAgent(agent_name="")


async def test_stub_yields_token_then_structured() -> None:
    stub = StubAgent(agent_name="market_research")
    events = await _collect(
        stub, _classification("market_research", entities={"tickers": ["NVDA"]})
    )
    types = [e.type for e in events]
    assert types == ["token", "structured"]
    assert isinstance(events[1].data, StubResponse)


async def test_stub_preserves_classifier_output() -> None:
    """Intent, agent, and ALL extracted entities are forwarded verbatim."""
    cls = _classification(
        "financial_calculator",
        intent="future_value",
        entities={
            "tickers": [],
            "topics": ["dca"],
            "amount": 500.0,
            "rate": 0.07,
            "period_years": 30.0,
        },
    )
    stub = StubAgent(agent_name="financial_calculator")
    events = await _collect(stub, cls)
    payload = events[-1].data
    assert isinstance(payload, StubResponse)
    assert payload.intent == "future_value"
    assert payload.agent == "financial_calculator"
    assert payload.status == "not_implemented"
    assert payload.entities.amount == 500.0
    assert payload.entities.rate == 0.07
    assert payload.entities.period_years == 30.0
    assert "dca" in payload.entities.topics


async def test_stub_message_mentions_agent_name_and_purpose() -> None:
    stub = StubAgent(agent_name="risk_assessment")
    events = await _collect(stub, _classification("risk_assessment"))
    text = (events[0].text or "")
    assert "risk_assessment" in text
    assert "isn't implemented" in text
    # Per-agent purpose summary is present.
    assert any(
        word in text.lower()
        for word in ("var", "drawdown", "stress", "exposure")
    )


async def test_stub_works_for_unknown_agent_name() -> None:
    stub = StubAgent(agent_name="future_specialist_xyz")
    events = await _collect(stub, _classification("future_specialist_xyz"))
    assert events[0].type == "token"
    payload = events[-1].data
    assert isinstance(payload, StubResponse)
    assert payload.agent == "future_specialist_xyz"


async def test_stub_handles_empty_entities() -> None:
    stub = StubAgent(agent_name="support")
    events = await _collect(stub, _classification("support"))
    text = events[0].text or ""
    # No "I captured: ..." clause when there are no entities.
    assert "captured" not in text.lower()


async def test_stub_summarises_entities_in_prose() -> None:
    cls = _classification("market_research", entities={"tickers": ["NVDA", "AMD"]})
    stub = StubAgent(agent_name="market_research")
    events = await _collect(stub, cls)
    text = events[0].text or ""
    assert "NVDA" in text
    assert "AMD" in text


# ---------------------------------------------------------------------------
# AgentRegistry
# ---------------------------------------------------------------------------

def test_registry_requires_non_empty_taxonomy() -> None:
    with pytest.raises(ValueError):
        AgentRegistry(taxonomy=[])


def test_registry_rejects_implemented_outside_taxonomy(market_data) -> None:
    """Implemented agent that's not in the taxonomy is almost always a bug
    (it would never get routed to). Rejecting at construction surfaces it
    immediately."""
    with pytest.raises(ValueError, match="not in taxonomy"):
        AgentRegistry(
            taxonomy=["portfolio_health"],
            implemented={"some_unknown_agent": PortfolioHealthAgent(market_data=market_data)},
        )


def test_registry_covers_every_taxonomy_agent(taxonomy: list[str], market_data) -> None:
    registry = build_default_registry(taxonomy=taxonomy, market_data=market_data)
    assert set(registry.names) == set(taxonomy)
    for name in taxonomy:
        agent = registry.get(name)
        assert isinstance(agent, BaseAgent)


def test_registry_implemented_set(taxonomy: list[str], market_data) -> None:
    registry = build_default_registry(taxonomy=taxonomy, market_data=market_data)
    assert "portfolio_health" in registry.implemented
    assert registry.is_implemented("portfolio_health")
    assert not registry.is_implemented("market_research")


def test_registry_unknown_name_falls_back_to_stub(taxonomy: list[str], market_data) -> None:
    """A misclassified agent name (LLM regression, taxonomy drift) must
    still produce a typed response — the router is the safety net."""
    registry = build_default_registry(taxonomy=taxonomy, market_data=market_data)
    agent = registry.get("definitely_not_a_real_agent_v9")
    assert isinstance(agent, StubAgent)


async def test_every_taxonomy_agent_streams_without_crashing(
    taxonomy: list[str], market_data
) -> None:
    """End-to-end: every taxonomy name routes through the registry and
    produces (token + structured) events without raising."""
    registry = build_default_registry(taxonomy=taxonomy, market_data=market_data)
    for name in taxonomy:
        agent = registry.get(name)
        cls = _classification(name)
        # portfolio_health needs a user; pass None and accept the error event,
        # which is the documented contract.
        events: list[AgentEvent] = []
        async for ev in agent.stream(classification=cls, user=None):
            events.append(ev)
        # At minimum we get one event of some kind — never zero, never raise.
        assert events
        assert all(isinstance(e, AgentEvent) for e in events)


async def test_default_registry_routes_portfolio_health_to_real_agent(
    taxonomy: list[str], market_data
) -> None:
    registry = build_default_registry(taxonomy=taxonomy, market_data=market_data)
    agent = registry.get("portfolio_health")
    assert isinstance(agent, PortfolioHealthAgent)


# ---------------------------------------------------------------------------
# Extensibility — adding a new agent later
# ---------------------------------------------------------------------------

async def test_registry_extension_is_a_one_liner(taxonomy: list[str], market_data) -> None:
    """Demonstrate that adding a new specialist tomorrow is one entry in the
    `implemented` dict — no code changes elsewhere."""

    class FakeNewAgent(BaseAgent):
        name = "market_research"

        async def stream(self, *, classification, user=None, history=None):
            yield AgentEvent(type="token", text="hello from the new specialist")

    registry = AgentRegistry(
        taxonomy=taxonomy,
        implemented={
            "portfolio_health": PortfolioHealthAgent(market_data=market_data),
            "market_research": FakeNewAgent(),
        },
    )
    assert registry.is_implemented("market_research")
    agent = registry.get("market_research")
    events = await _collect(agent, _classification("market_research"))
    assert events[0].text == "hello from the new specialist"
