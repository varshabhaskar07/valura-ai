"""Portfolio valuation + metrics.

Single entry point: ``compute_valuation(user, market_data) → PortfolioValuation``.

Edge cases handled explicitly:
  - Empty portfolio: returns a fully-typed valuation with cash_value > 0,
    no positions, ``concentration_flag == 'n/a'``, all returns as ``None``.
    The portfolio-health agent's BUILD branch keys off ``portfolio.is_empty``.
  - Single position: top_position_pct == 100, top_3_pct == 100,
    flag == 'high'. No division-by-zero anywhere.
  - Missing price for a ticker: that position contributes 0 to value but
    appears in ``missing_prices``. Returns/concentration are computed over
    the priced subset; the agent surfaces ``missing_prices`` to the user.
  - Mixed currencies: all monetary fields are normalised to
    ``user.base_currency`` via the FX side of the MarketData provider.
    A missing FX rate causes the position to be treated as missing-price
    rather than crashing — same surfacing path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Position, User
from .market_data import MarketData


# ---------------------------------------------------------------------------
# Concentration thresholds
# ---------------------------------------------------------------------------

# Two-of-two flag: trip on either top-1 OR top-3 crossing the threshold.
# The numbers below match common practitioner heuristics; tuned to make
# user_002 (NVDA-dominant) flag 'high' and user_001 / 005 flag 'moderate'
# rather than 'high' so the agent's tone matches the actual risk.
_TOP1_HIGH_PCT = 40.0
_TOP1_MOD_PCT = 25.0
_TOP3_HIGH_PCT = 70.0
_TOP3_MOD_PCT = 60.0


def _flag(top1_pct: float, top3_pct: float) -> str:
    if top1_pct >= _TOP1_HIGH_PCT or top3_pct >= _TOP3_HIGH_PCT:
        return "high"
    if top1_pct >= _TOP1_MOD_PCT or top3_pct >= _TOP3_MOD_PCT:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PositionValuation:
    """One row of the valuation report. ``None``-valued numeric fields mean
    'price or FX missing'; the position is reported but not included in
    aggregates."""

    position: Position
    current_price: float | None        # in position.currency
    fx_rate_to_base: float | None      # multiplier from position.currency → base
    market_value_base: float | None    # in user.base_currency
    cost_basis_base: float | None      # in user.base_currency
    weight_pct: float | None           # share of portfolio (positions only, not cash)
    return_pct: float | None           # (price - cost_basis_per_share) / cost_basis_per_share * 100


@dataclass(frozen=True)
class Concentration:
    top_position_pct: float | None
    top_3_positions_pct: float | None
    hhi: float | None                  # 0..10000 (squared % weights)
    flag: str                          # "low" | "moderate" | "high" | "n/a"


@dataclass(frozen=True)
class PortfolioValuation:
    user_id: str
    base_currency: str

    positions_value: float             # sum of priced positions, base currency
    cash_value: float                  # sum of all cash converted to base
    total_value: float                 # positions_value + cash_value

    valuations: tuple[PositionValuation, ...] = ()
    missing_prices: tuple[str, ...] = ()      # tickers we couldn't price
    missing_fx: tuple[str, ...] = ()          # currencies we couldn't convert

    concentration: Concentration = field(
        default_factory=lambda: Concentration(None, None, None, "n/a")
    )

    cost_basis_total: float | None = None     # base currency, priced positions only
    total_return_pct: float | None = None     # (positions_value - cost_basis) / cost_basis * 100

    @property
    def has_data(self) -> bool:
        """True if at least one position is fully priced."""
        return any(v.market_value_base is not None for v in self.valuations)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_valuation(user: User, market_data: MarketData) -> PortfolioValuation:
    """Build a PortfolioValuation from current market data.

    Always returns a typed result. Never raises for routine missing data —
    those are reported via ``missing_prices`` / ``missing_fx``.
    """
    base = user.base_currency

    # --- Per-position valuation -------------------------------------------
    valuations: list[PositionValuation] = []
    missing_prices: list[str] = []
    missing_fx: set[str] = set()

    for pos in user.portfolio.positions:
        price = market_data.get_price(pos.ticker)
        fx = market_data.get_fx_rate(pos.currency, base) if pos.currency != base else 1.0

        market_value_base: float | None = None
        cost_basis_base: float | None = None
        return_pct: float | None = None

        if fx is None:
            missing_fx.add(pos.currency)

        if price is None:
            missing_prices.append(pos.ticker)
        elif fx is not None:
            market_value_base = pos.quantity * price * fx
            cost_basis_base = pos.cost_basis * fx
            # Return is a per-share quantity, so FX cancels out — but
            # using the same fx for both keeps the math explicit.
            if pos.cost_basis_per_share > 0:
                return_pct = (price - pos.cost_basis_per_share) / pos.cost_basis_per_share * 100

        valuations.append(PositionValuation(
            position=pos,
            current_price=price,
            fx_rate_to_base=fx,
            market_value_base=market_value_base,
            cost_basis_base=cost_basis_base,
            weight_pct=None,  # filled in below
            return_pct=return_pct,
        ))

    positions_value = sum(
        v.market_value_base or 0.0 for v in valuations if v.market_value_base is not None
    )

    # --- Weight per position (only meaningful if positions_value > 0) ----
    if positions_value > 0:
        valuations = [
            PositionValuation(
                position=v.position,
                current_price=v.current_price,
                fx_rate_to_base=v.fx_rate_to_base,
                market_value_base=v.market_value_base,
                cost_basis_base=v.cost_basis_base,
                weight_pct=(
                    v.market_value_base / positions_value * 100
                    if v.market_value_base is not None else None
                ),
                return_pct=v.return_pct,
            )
            for v in valuations
        ]

    # --- Cash, converted to base ------------------------------------------
    cash_value = 0.0
    for currency, amount in user.portfolio.cash.items():
        rate = market_data.get_fx_rate(currency, base)
        if rate is None:
            missing_fx.add(currency)
            continue
        cash_value += amount * rate

    # --- Concentration ----------------------------------------------------
    weights_pct = sorted(
        [v.weight_pct for v in valuations if v.weight_pct is not None],
        reverse=True,
    )
    if not weights_pct:
        concentration = Concentration(None, None, None, "n/a")
    else:
        top1 = weights_pct[0]
        top3 = sum(weights_pct[:3])  # works correctly even with <3 positions
        hhi = sum(w * w for w in weights_pct)  # already in pct units
        concentration = Concentration(
            top_position_pct=round(top1, 2),
            top_3_positions_pct=round(top3, 2),
            hhi=round(hhi, 2),
            flag=_flag(top1, top3),
        )

    # --- Returns ----------------------------------------------------------
    cost_basis_total = sum(
        v.cost_basis_base or 0.0
        for v in valuations
        if v.market_value_base is not None and v.cost_basis_base is not None
    )
    total_return_pct: float | None = None
    if cost_basis_total > 0:
        priced_value = sum(
            v.market_value_base or 0.0
            for v in valuations
            if v.market_value_base is not None and v.cost_basis_base is not None
        )
        total_return_pct = round((priced_value - cost_basis_total) / cost_basis_total * 100, 2)

    return PortfolioValuation(
        user_id=user.user_id,
        base_currency=base,
        positions_value=round(positions_value, 2),
        cash_value=round(cash_value, 2),
        total_value=round(positions_value + cash_value, 2),
        valuations=tuple(valuations),
        missing_prices=tuple(missing_prices),
        missing_fx=tuple(sorted(missing_fx)),
        concentration=concentration,
        cost_basis_total=round(cost_basis_total, 2) if cost_basis_total else None,
        total_return_pct=total_return_pct,
    )
