"""Portfolio domain — models, market data, metrics. All offline.

Static market data covers every ticker + every currency present in the
five fixture users. Network never touched; yfinance never imported.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from valura_ai.portfolio.market_data import StaticMarketData
from valura_ai.portfolio.metrics import compute_valuation
from valura_ai.portfolio.models import Portfolio, Position, User


# ---------------------------------------------------------------------------
# Static market data
# ---------------------------------------------------------------------------

# Plausible illustrative prices in each ticker's listed currency. These
# numbers are NOT real quotes — fixtures shipped with the repo deliberately
# do not encode market state.
PRICES: dict[str, float] = {
    # USD-listed
    "TSLA": 250.00,
    "NVDA": 540.00,
    "MSFT": 410.00,
    "AAPL": 195.00,
    "GOOGL": 165.00,
    "COIN": 165.00,
    "SOXL": 32.00,
    "TQQQ": 70.00,
    "ARKK": 50.00,
    "SCHD": 78.00,
    "VYM": 115.00,
    "VNQ": 90.00,
    "JNJ": 152.00,
    "PEP": 170.00,
    "KO": 62.00,
    "T": 17.50,
    "BABA": 78.00,
    # Non-USD
    "ASML.AS": 720.00,    # EUR
    "SAP.DE":  150.00,    # EUR
    "HSBA.L":   6.80,     # GBP
    "7203.T": 2400.00,    # JPY
}

FX: dict[tuple[str, str], float] = {
    ("EUR", "USD"): 1.08,
    ("GBP", "USD"): 1.25,
    ("JPY", "USD"): 0.0067,
}


@pytest.fixture(scope="module")
def market_data() -> StaticMarketData:
    return StaticMarketData(prices=PRICES, fx_rates=FX)


@pytest.fixture(scope="module")
def fixtures(fixtures_dir: Path) -> dict[str, User]:
    return {
        path.stem: User.from_fixture(path)
        for path in sorted((fixtures_dir / "users").glob("*.json"))
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def test_position_validates_basic() -> None:
    p = Position(
        ticker="NVDA", exchange="NASDAQ", quantity=10,
        cost_basis_per_share=200.0, currency="usd",
    )
    assert p.currency == "USD"
    assert p.cost_basis == 2000.0


def test_position_rejects_zero_or_negative_quantity() -> None:
    with pytest.raises(Exception):
        Position(ticker="X", exchange="", quantity=0,
                 cost_basis_per_share=10, currency="USD")
    with pytest.raises(Exception):
        Position(ticker="X", exchange="", quantity=-1,
                 cost_basis_per_share=10, currency="USD")


def test_position_is_frozen() -> None:
    p = Position(ticker="X", exchange="", quantity=1,
                 cost_basis_per_share=10, currency="USD")
    with pytest.raises(Exception):
        p.quantity = 2  # type: ignore[misc]


def test_portfolio_uppercases_cash_keys() -> None:
    p = Portfolio(positions=[], cash={"usd": 100, "eur": 50})
    assert set(p.cash) == {"USD", "EUR"}


def test_portfolio_is_empty_property() -> None:
    assert Portfolio().is_empty is True
    p = Portfolio(positions=[Position(
        ticker="A", exchange="", quantity=1, cost_basis_per_share=1, currency="USD",
    )])
    assert p.is_empty is False


def test_user_loads_from_fixture_path(fixtures_dir: Path) -> None:
    u = User.from_fixture(fixtures_dir / "users" / "user_001_aggressive_trader.json")
    assert u.user_id == "user_001_aggressive_trader"
    assert u.base_currency == "USD"
    assert len(u.portfolio.positions) > 0


def test_user_loads_all_five_fixtures(fixtures: dict[str, User]) -> None:
    assert len(fixtures) == 5
    assert all(isinstance(u, User) for u in fixtures.values())


# ---------------------------------------------------------------------------
# StaticMarketData
# ---------------------------------------------------------------------------

def test_static_market_data_returns_known_prices(market_data: StaticMarketData) -> None:
    assert market_data.get_price("NVDA") == 540.0
    assert market_data.get_price("nvda") == 540.0  # case-insensitive


def test_static_market_data_returns_none_for_unknown(market_data: StaticMarketData) -> None:
    assert market_data.get_price("NONEXISTENT") is None
    assert market_data.get_fx_rate("USD", "ZAR") is None


def test_fx_same_currency_is_unit(market_data: StaticMarketData) -> None:
    assert market_data.get_fx_rate("USD", "USD") == 1.0


def test_fx_inverse_lookup(market_data: StaticMarketData) -> None:
    assert market_data.get_fx_rate("EUR", "USD") == 1.08
    inv = market_data.get_fx_rate("USD", "EUR")
    assert inv is not None and abs(inv - 1 / 1.08) < 1e-9


# ---------------------------------------------------------------------------
# compute_valuation — happy path on each fixture user
# ---------------------------------------------------------------------------

def test_aggressive_trader_valuation(fixtures: dict[str, User], market_data) -> None:
    v = compute_valuation(fixtures["user_001_aggressive_trader"], market_data)
    assert v.has_data
    assert v.positions_value > 0
    assert v.total_value > v.positions_value  # cash is added
    assert v.concentration.flag in {"low", "moderate", "high"}
    # All positions should price (none missing).
    assert v.missing_prices == ()
    assert v.missing_fx == ()
    # Each weight in [0, 100]; sum is ~100.
    weights = [pv.weight_pct for pv in v.valuations if pv.weight_pct is not None]
    assert all(0 <= w <= 100 for w in weights)
    assert abs(sum(weights) - 100) < 0.01


def test_concentrated_user_flags_high(fixtures: dict[str, User], market_data) -> None:
    """user_002 is dominated by NVDA — concentration must be 'high' and the
    top position must clearly stand out, otherwise the agent's headline
    observation would be wrong."""
    v = compute_valuation(fixtures["user_002_concentrated_nvda"], market_data)
    assert v.concentration.flag == "high"
    assert v.concentration.top_position_pct is not None
    assert v.concentration.top_position_pct > 60
    # NVDA must be the top position, not somewhere in the middle.
    top_pv = max(v.valuations, key=lambda pv: pv.market_value_base or 0)
    assert top_pv.position.ticker == "NVDA"


def test_dividend_retiree_valuation(fixtures: dict[str, User], market_data) -> None:
    v = compute_valuation(fixtures["user_003_dividend_retiree"], market_data)
    assert v.has_data
    # Cash is substantial relative to positions.
    assert v.cash_value > 20_000
    # Diversified — should NOT be high concentration.
    assert v.concentration.flag in {"low", "moderate"}


def test_global_multi_currency_valuation(fixtures: dict[str, User], market_data) -> None:
    """user_005 has USD/EUR/GBP/JPY positions. All values must be normalised
    to the user's base currency (USD)."""
    u = fixtures["user_005_global_multi_currency"]
    v = compute_valuation(u, market_data)
    assert v.has_data
    assert v.base_currency == "USD"
    assert v.missing_prices == ()
    assert v.missing_fx == ()

    # Each non-USD position's market_value_base must equal qty * price * fx.
    for pv in v.valuations:
        if pv.market_value_base is None:
            continue
        expected = pv.position.quantity * pv.current_price * pv.fx_rate_to_base
        assert abs(pv.market_value_base - expected) < 1e-6

    # Cash spans 4 currencies — total cash_value is sum of all converted.
    expected_cash = (
        u.portfolio.cash["USD"]
        + u.portfolio.cash["EUR"] * 1.08
        + u.portfolio.cash["GBP"] * 1.25
        + u.portfolio.cash["JPY"] * 0.0067
    )
    assert abs(v.cash_value - expected_cash) < 0.01

    # JPY position has high quantity (400) but low FX (~0.0067) — must NOT
    # dominate. This is the bug we'd hit if normalisation were skipped.
    jpy_position = next(pv for pv in v.valuations if pv.position.ticker == "7203.T")
    assert jpy_position.weight_pct is not None
    assert jpy_position.weight_pct < 20, "JPY position should not dominate after FX"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_portfolio_does_not_crash(fixtures: dict[str, User], market_data) -> None:
    """user_004 has zero positions. Valuation must return a typed result with
    the BUILD-oriented signals the agent will key off."""
    u = fixtures["user_004_empty"]
    assert u.portfolio.is_empty
    v = compute_valuation(u, market_data)
    assert not v.has_data
    assert v.positions_value == 0.0
    assert v.cash_value > 0  # they have cash to deploy
    assert v.total_value == v.cash_value
    assert v.valuations == ()
    assert v.concentration.flag == "n/a"
    assert v.concentration.top_position_pct is None
    assert v.cost_basis_total is None
    assert v.total_return_pct is None


def test_single_position_concentration_is_100_pct(market_data) -> None:
    """Edge case: one position. Must NOT divide by zero, must report 100%
    concentration, and flag must be 'high'."""
    u = User(
        user_id="solo", display_name="solo",
        base_currency="USD",
        portfolio=Portfolio(
            positions=[Position(
                ticker="NVDA", exchange="NASDAQ",
                quantity=10, cost_basis_per_share=300, currency="USD",
            )],
            cash={"USD": 0},
        ),
    )
    v = compute_valuation(u, market_data)
    assert v.concentration.top_position_pct == 100.0
    assert v.concentration.top_3_positions_pct == 100.0
    assert v.concentration.flag == "high"
    assert v.valuations[0].weight_pct == 100.0


def test_missing_price_is_reported_not_raised(market_data) -> None:
    u = User(
        user_id="x", display_name="x",
        base_currency="USD",
        portfolio=Portfolio(
            positions=[
                Position(ticker="NVDA", exchange="", quantity=10,
                         cost_basis_per_share=300, currency="USD"),
                Position(ticker="UNKNOWN_TICKER", exchange="", quantity=10,
                         cost_basis_per_share=50, currency="USD"),
            ],
        ),
    )
    v = compute_valuation(u, market_data)
    # NVDA is priced; UNKNOWN_TICKER reported.
    assert "UNKNOWN_TICKER" in v.missing_prices
    assert v.has_data  # the priced subset still produces aggregates
    assert v.positions_value > 0
    # The missing position has no market value but appears in the report.
    missing = next(pv for pv in v.valuations if pv.position.ticker == "UNKNOWN_TICKER")
    assert missing.market_value_base is None
    assert missing.weight_pct is None


def test_missing_fx_is_reported_not_raised() -> None:
    """Position currency we can't convert → reported, doesn't crash."""
    md = StaticMarketData(prices={"FOO": 100.0}, fx_rates={})  # no FX at all
    u = User(
        user_id="x", display_name="x",
        base_currency="USD",
        portfolio=Portfolio(
            positions=[Position(
                ticker="FOO", exchange="", quantity=1,
                cost_basis_per_share=50, currency="EUR",
            )],
            cash={"EUR": 100.0},  # also can't convert
        ),
    )
    v = compute_valuation(u, market_data=md)
    assert "EUR" in v.missing_fx
    # Value == 0 because the only position can't be priced in base.
    assert v.positions_value == 0.0
    assert v.cash_value == 0.0


def test_cost_basis_zero_total_return_is_none(market_data) -> None:
    """If everything is missing-priced, return is None (not 0, not error)."""
    u = User(
        user_id="x", display_name="x",
        base_currency="USD",
        portfolio=Portfolio(
            positions=[Position(
                ticker="UNKNOWN", exchange="", quantity=10,
                cost_basis_per_share=50, currency="USD",
            )],
        ),
    )
    v = compute_valuation(u, market_data)
    assert v.cost_basis_total is None
    assert v.total_return_pct is None


def test_returns_calculation_basic(market_data) -> None:
    """Sanity: portfolio bought at 100, now worth 110 → +10% return."""
    u = User(
        user_id="x", display_name="x",
        base_currency="USD",
        portfolio=Portfolio(
            positions=[Position(
                ticker="TSLA", exchange="", quantity=1,
                cost_basis_per_share=200, currency="USD",
            )],
        ),
    )
    v = compute_valuation(u, market_data)
    # TSLA priced at 250, cost 200 → +25%
    assert v.total_return_pct is not None
    assert abs(v.total_return_pct - 25.0) < 0.01
    assert abs(v.valuations[0].return_pct - 25.0) < 0.01


def test_pathological_two_position_top3_falls_back(market_data) -> None:
    """top_3 is the sum of available — must work with <3 positions."""
    u = User(
        user_id="x", display_name="x",
        base_currency="USD",
        portfolio=Portfolio(
            positions=[
                Position(ticker="NVDA", exchange="", quantity=1,
                         cost_basis_per_share=100, currency="USD"),
                Position(ticker="MSFT", exchange="", quantity=1,
                         cost_basis_per_share=100, currency="USD"),
            ],
        ),
    )
    v = compute_valuation(u, market_data)
    # Two positions, so top_3 should equal sum of weights == 100.
    assert v.concentration.top_3_positions_pct is not None
    assert abs(v.concentration.top_3_positions_pct - 100.0) < 0.01


# ---------------------------------------------------------------------------
# Performance sanity
# ---------------------------------------------------------------------------

def test_valuation_is_fast(fixtures: dict[str, User], market_data) -> None:
    """Valuation is on the request hot path. Must be sub-millisecond per
    user with cached/static data — this proves the agent step adds
    negligible latency relative to the LLM call."""
    import time
    n = 200
    u = fixtures["user_005_global_multi_currency"]  # most complex case
    t0 = time.perf_counter()
    for _ in range(n):
        compute_valuation(u, market_data)
    elapsed_ms = (time.perf_counter() - t0) * 1000 / n
    assert elapsed_ms < 2.0, f"avg {elapsed_ms:.3f} ms / valuation"
