"""Portfolio Health agent tests.

Covers:
  - The three required user shapes (high-concentration, empty, multi-currency).
  - Streaming contract (token events come BEFORE the structured event).
  - Disclaimer present on every response.
  - Missing data is surfaced as observation, not raised.
  - First-token latency is well under the assignment's 2s budget.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from valura_ai.agents.base import AgentEvent
from valura_ai.agents.portfolio_health import (
    DISCLAIMER,
    LEVERAGED_TICKERS,
    PortfolioHealthAgent,
)
from valura_ai.agents.schemas import PortfolioHealthReport
from valura_ai.classifier.schema import ClassificationResult
from valura_ai.portfolio.market_data import StaticMarketData
from valura_ai.portfolio.models import Portfolio, Position, User


# Same prices/FX the portfolio tests use.
PRICES: dict[str, float] = {
    "TSLA": 250, "NVDA": 540, "MSFT": 410, "AAPL": 195, "GOOGL": 165, "COIN": 165,
    "SOXL": 32, "TQQQ": 70, "ARKK": 50, "SCHD": 78, "VYM": 115, "VNQ": 90,
    "JNJ": 152, "PEP": 170, "KO": 62, "T": 17.50, "BABA": 78,
    "ASML.AS": 720, "SAP.DE": 150, "HSBA.L": 6.80, "7203.T": 2400,
}
FX: dict[tuple[str, str], float] = {
    ("EUR", "USD"): 1.08, ("GBP", "USD"): 1.25, ("JPY", "USD"): 0.0067,
}


@pytest.fixture(scope="module")
def market_data() -> StaticMarketData:
    return StaticMarketData(prices=PRICES, fx_rates=FX)


@pytest.fixture(scope="module")
def agent(market_data: StaticMarketData) -> PortfolioHealthAgent:
    return PortfolioHealthAgent(market_data=market_data)


@pytest.fixture(scope="module")
def fixtures(fixtures_dir: Path) -> dict[str, User]:
    return {
        path.stem: User.from_fixture(path)
        for path in sorted((fixtures_dir / "users").glob("*.json"))
    }


def _classification(agent_name: str = "portfolio_health") -> ClassificationResult:
    return ClassificationResult(intent="health_check", agent=agent_name)


async def _collect(agent: PortfolioHealthAgent, user: User) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    async for ev in agent.stream(classification=_classification(), user=user):
        events.append(ev)
    return events


def _prose(events: list[AgentEvent]) -> str:
    return "".join(e.text or "" for e in events if e.type == "token")


def _structured(events: list[AgentEvent]) -> PortfolioHealthReport | None:
    for e in events:
        if e.type == "structured" and isinstance(e.data, PortfolioHealthReport):
            return e.data
    return None


# ---------------------------------------------------------------------------
# Streaming contract
# ---------------------------------------------------------------------------

async def test_stream_yields_tokens_before_structured(
    agent: PortfolioHealthAgent, fixtures: dict[str, User]
) -> None:
    events = await _collect(agent, fixtures["user_001_aggressive_trader"])
    types = [e.type for e in events]
    # At least one token event before the (single) structured event.
    assert "token" in types
    assert types.count("structured") == 1
    structured_idx = types.index("structured")
    assert any(t == "token" for t in types[:structured_idx])


async def test_no_user_yields_error_event(agent: PortfolioHealthAgent) -> None:
    events: list[AgentEvent] = []
    async for ev in agent.stream(classification=_classification(), user=None):
        events.append(ev)
    assert len(events) == 1
    assert events[0].type == "error"
    assert "user context" in (events[0].text or "").lower()


async def test_first_token_latency(
    agent: PortfolioHealthAgent, fixtures: dict[str, User]
) -> None:
    """First-token latency budget is 2s; deterministic agent should be ~ms."""
    import time
    user = fixtures["user_005_global_multi_currency"]
    t0 = time.perf_counter()
    async for _ev in agent.stream(classification=_classification(), user=user):
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 100, f"first token took {elapsed_ms:.1f} ms"
        break


# ---------------------------------------------------------------------------
# Required user shapes
# ---------------------------------------------------------------------------

async def test_high_concentration_user_002(
    agent: PortfolioHealthAgent, fixtures: dict[str, User]
) -> None:
    events = await _collect(agent, fixtures["user_002_concentrated_nvda"])
    report = _structured(events)
    assert report is not None
    assert report.concentration_risk.flag == "high"
    assert report.concentration_risk.top_position_pct is not None
    assert report.concentration_risk.top_position_pct > 60
    assert not report.is_build_oriented

    # Must mention NVDA by name, percent, and the trim-suggestion line.
    prose = _prose(events).lower()
    assert "nvda" in prose
    assert "%" in prose
    assert "trim" in prose or "trimming" in prose

    # Disclaimer always present.
    assert DISCLAIMER in _prose(events)
    assert report.disclaimer == DISCLAIMER

    # The headline observation should be the concentration warning.
    assert report.observations
    headline_obs = report.observations[0]
    assert headline_obs.severity == "warning"
    assert "nvda" in headline_obs.text.lower()


async def test_empty_portfolio_user_004_build_branch(
    agent: PortfolioHealthAgent, fixtures: dict[str, User]
) -> None:
    """user_004 is the must-not-crash case. Output must be BUILD-oriented."""
    user = fixtures["user_004_empty"]
    events = await _collect(agent, user)
    report = _structured(events)

    assert report is not None
    assert report.is_build_oriented is True
    # No metrics are claimed.
    assert report.concentration_risk.flag == "n/a"
    assert report.performance.total_return_pct is None
    assert report.benchmark_comparison is None

    # Headline is forward-looking, not "no data".
    assert "start line" in report.headline.lower() or "ready to deploy" in report.headline.lower()

    # Prose mentions cash amount and gives a starter direction.
    prose = _prose(events)
    assert "$10,000" in prose or "10,000" in prose
    assert any(token in prose.lower() for token in ("vti", "vxus", "bnd", "etf", "allocation"))
    assert DISCLAIMER in prose


async def test_multi_currency_user_005(
    agent: PortfolioHealthAgent, fixtures: dict[str, User]
) -> None:
    events = await _collect(agent, fixtures["user_005_global_multi_currency"])
    report = _structured(events)

    assert report is not None
    # Currency normalisation upstream — the report uses USD throughout, no
    # raw JPY/EUR amounts in prose.
    prose = _prose(events)
    assert "$" in prose
    # JPY position should NOT dominate (it would in raw quantity terms).
    assert report.concentration_risk.flag in {"low", "moderate"}
    # No missing-data warnings — prices and FX cover the whole book.
    assert report.missing_data.tickers == []
    assert report.missing_data.currencies == []


# ---------------------------------------------------------------------------
# Other shapes — not strictly required, but cheap to pin
# ---------------------------------------------------------------------------

async def test_aggressive_user_001_leverage_observation(
    agent: PortfolioHealthAgent, fixtures: dict[str, User]
) -> None:
    """Aggressive user holds TQQQ + SOXL — leverage warning must appear,
    framed neutrally because the user opted into aggressive."""
    events = await _collect(agent, fixtures["user_001_aggressive_trader"])
    report = _structured(events)
    assert report is not None
    obs_text = " ".join(o.text for o in report.observations).lower()
    assert "leveraged" in obs_text or "tqqq" in obs_text or "soxl" in obs_text


async def test_dividend_retiree_user_003_no_lecture(
    agent: PortfolioHealthAgent, fixtures: dict[str, User]
) -> None:
    """Conservative income holder — no high-concentration warning, no
    pushy 'add risk' suggestions."""
    events = await _collect(agent, fixtures["user_003_dividend_retiree"])
    report = _structured(events)
    assert report is not None
    assert report.concentration_risk.flag in {"low", "moderate"}
    prose = _prose(events).lower()
    # No prescriptive language about adding risk to an income portfolio.
    assert "you should buy" not in prose
    assert "must add" not in prose


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

async def test_missing_price_is_surfaced_not_raised(market_data: StaticMarketData) -> None:
    """Missing price for a held ticker must appear as a warning observation,
    not crash the stream."""
    md = StaticMarketData(prices={"NVDA": 540, "MSFT": 410}, fx_rates={})
    agent = PortfolioHealthAgent(market_data=md)
    user = User(
        user_id="x", display_name="x", base_currency="USD",
        portfolio=Portfolio(
            positions=[
                Position(ticker="NVDA", exchange="", quantity=10,
                         cost_basis_per_share=300, currency="USD"),
                Position(ticker="UNKNOWN", exchange="", quantity=5,
                         cost_basis_per_share=50, currency="USD"),
            ],
        ),
    )
    events = await _collect(agent, user)
    report = _structured(events)
    assert report is not None
    assert "UNKNOWN" in report.missing_data.tickers
    obs_text = " ".join(o.text for o in report.observations).lower()
    assert "couldn't price" in obs_text or "missing" in obs_text


async def test_missing_fx_is_surfaced_not_raised() -> None:
    """A position in a currency we can't convert — surface, don't crash."""
    md = StaticMarketData(prices={"FOO": 100.0, "NVDA": 540.0}, fx_rates={})
    agent = PortfolioHealthAgent(market_data=md)
    user = User(
        user_id="x", display_name="x", base_currency="USD",
        portfolio=Portfolio(
            positions=[
                Position(ticker="NVDA", exchange="", quantity=10,
                         cost_basis_per_share=300, currency="USD"),
                Position(ticker="FOO", exchange="", quantity=10,
                         cost_basis_per_share=80, currency="EUR"),
            ],
        ),
    )
    events = await _collect(agent, user)
    report = _structured(events)
    assert report is not None
    assert "EUR" in report.missing_data.currencies


async def test_observations_capped_at_four(
    agent: PortfolioHealthAgent, fixtures: dict[str, User]
) -> None:
    """Assignment: 'surface the one or two things that matter most rather
    than dumping every metric'. We cap at 4."""
    for user in fixtures.values():
        events = await _collect(agent, user)
        report = _structured(events)
        assert report is not None
        assert len(report.observations) <= 4


async def test_disclaimer_on_every_response(
    agent: PortfolioHealthAgent, fixtures: dict[str, User]
) -> None:
    for user in fixtures.values():
        events = await _collect(agent, user)
        report = _structured(events)
        assert report is not None
        assert report.disclaimer == DISCLAIMER
        assert DISCLAIMER in _prose(events)


async def test_known_leveraged_tickers_set_includes_user_001_holdings() -> None:
    # Sanity: the user_001 fixture asserts leverage warning fires; that
    # depends on the constant including those tickers.
    assert "TQQQ" in LEVERAGED_TICKERS
    assert "SOXL" in LEVERAGED_TICKERS
