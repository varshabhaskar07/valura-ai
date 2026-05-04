"""Portfolio Health agent — the implemented specialist for the assignment.

Streams a short, novice-friendly health check, then emits the structured
``PortfolioHealthReport``. No LLM call: the data IS the answer here, so
templated prose keeps first-token latency at zero, makes the agent fully
testable offline, and avoids drift between the prose and the structured
fields a downstream consumer reads.

Branching:
  - empty portfolio       → BUILD branch (different headline, no metrics)
  - high concentration    → headline + warning observation up top
  - moderate              → headline + balanced framing
  - low / diversified     → affirmation + one nuance observation
  - aggressive risk + leveraged ETFs detected → leverage warning added

The two observations the assignment asks for ("the one or two things
that matter most") are picked by severity priority:
  critical > warning > info, then leverage > concentration > cash drag >
  performance > benchmark. Observations are capped at 4.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence
from typing import ClassVar

from ..classifier.schema import ClassificationResult
from ..llm.client import ChatMessage
from ..portfolio.market_data import MarketData
from ..portfolio.metrics import PortfolioValuation, compute_valuation
from ..portfolio.models import User
from .base import AgentEvent, BaseAgent
from .schemas import (
    BenchmarkComparison,
    ConcentrationRisk,
    HealthObservation,
    MissingData,
    PerformanceSummary,
    PortfolioHealthReport,
)


# Standard regulatory disclaimer — appended verbatim to every response.
DISCLAIMER = (
    "This is not investment advice. Information is provided for educational "
    "purposes only; consult a licensed financial advisor before making any "
    "investment decisions."
)

# Tickers we recognise as leveraged ETFs (2x or 3x daily) — used to add a
# leverage warning when present at meaningful weight.
LEVERAGED_TICKERS: frozenset[str] = frozenset({
    "TQQQ", "SQQQ", "SOXL", "SOXS", "UPRO", "SPXU", "TNA", "TZA",
    "FAS", "FAZ", "TMF", "TMV", "UDOW", "SDOW", "LABU", "LABD",
})

# Hardcoded benchmark approximations for the demo. We don't have time-series
# data, so we present these as "approximate" in the prose. Documented in the
# README's portfolio-health agent section.
BENCHMARK_BY_BASE_CURRENCY: dict[str, tuple[str, float]] = {
    "USD": ("S&P 500", 10.0),       # rough long-term annualized
    "EUR": ("STOXX 600", 7.0),
    "GBP": ("FTSE 100", 5.5),
    "JPY": ("TOPIX", 6.0),
}


class PortfolioHealthAgent(BaseAgent):
    """Implemented agent for portfolio_health intent."""

    name: ClassVar[str] = "portfolio_health"

    def __init__(self, *, market_data: MarketData) -> None:
        self._market_data = market_data

    async def stream(
        self,
        *,
        classification: ClassificationResult,
        user: User | None = None,
        history: Sequence[ChatMessage] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        if user is None:
            yield AgentEvent(
                type="error",
                text="Portfolio health requires user context — none was provided.",
            )
            return

        valuation = compute_valuation(user, self._market_data)
        report = self._build_report(user, valuation)

        # Stream the prose. Each chunk is one sentence — natural reading
        # cadence, and small enough that an SSE client renders progressively.
        for chunk in self._render_prose(user, report):
            yield AgentEvent(type="token", text=chunk + " ")

        # Always finish with the structured payload so downstream code
        # (history, telemetry, the SSE client's "result" handler) gets the
        # canonical fields, not just prose.
        yield AgentEvent(type="structured", data=report)

    # -----------------------------------------------------------------
    # Report construction
    # -----------------------------------------------------------------

    def _build_report(self, user: User, v: PortfolioValuation) -> PortfolioHealthReport:
        if user.portfolio.is_empty:
            return self._build_empty(user, v)

        return PortfolioHealthReport(
            user_id=user.user_id,
            headline=self._headline(user, v),
            concentration_risk=ConcentrationRisk(
                top_position_pct=v.concentration.top_position_pct,
                top_3_positions_pct=v.concentration.top_3_positions_pct,
                flag=v.concentration.flag,
            ),
            performance=PerformanceSummary(
                total_return_pct=v.total_return_pct,
                # Approximate: assume ~2y average holding period. Documented.
                annualized_return_pct=(
                    round(v.total_return_pct / 2, 2)
                    if v.total_return_pct is not None
                    else None
                ),
            ),
            benchmark_comparison=self._benchmark(user, v),
            observations=self._build_observations(user, v),
            missing_data=MissingData(
                tickers=list(v.missing_prices),
                currencies=list(v.missing_fx),
            ),
            is_build_oriented=False,
            disclaimer=DISCLAIMER,
        )

    def _build_empty(self, user: User, v: PortfolioValuation) -> PortfolioHealthReport:
        cash_str = self._fmt_amount(v.cash_value, user.base_currency)
        observations: list[HealthObservation] = [
            HealthObservation(
                severity="info",
                text=(
                    f"You have {cash_str} in cash and no positions yet. "
                    f"With a {user.risk_profile} risk profile, a typical first "
                    "allocation pairs broad equity exposure with some bond/cash "
                    "buffer."
                ),
            ),
            HealthObservation(
                severity="info",
                text=(
                    "Common starting points include a US total-market ETF "
                    "(e.g. VTI), an international ETF (VXUS), and bonds (BND) — "
                    "weighted to your risk tolerance."
                ),
            ),
        ]
        return PortfolioHealthReport(
            user_id=user.user_id,
            headline=f"You're at the start line — {cash_str} ready to deploy.",
            concentration_risk=ConcentrationRisk(flag="n/a"),
            performance=PerformanceSummary(),
            benchmark_comparison=None,
            observations=observations,
            missing_data=MissingData(),
            is_build_oriented=True,
            disclaimer=DISCLAIMER,
        )

    def _headline(self, user: User, v: PortfolioValuation) -> str:
        total = self._fmt_amount(v.total_value, user.base_currency)
        ret = (
            f", {self._fmt_pct(v.total_return_pct)} since cost basis"
            if v.total_return_pct is not None
            else ""
        )
        flag = v.concentration.flag
        flag_clause = (
            f" — concentration is {flag}" if flag in {"high", "moderate"} else ""
        )
        return f"Your portfolio is worth {total}{ret}{flag_clause}."

    def _benchmark(self, user: User, v: PortfolioValuation) -> BenchmarkComparison | None:
        if v.total_return_pct is None:
            return None
        bench = BENCHMARK_BY_BASE_CURRENCY.get(user.base_currency)
        if bench is None:
            return None
        name, bench_annualized_pct = bench
        # Compare apples-to-apples by approximating the portfolio's annualized
        # return using the same 2y assumption as PerformanceSummary above.
        port_annualized = v.total_return_pct / 2
        return BenchmarkComparison(
            benchmark=name,
            portfolio_return_pct=round(port_annualized, 2),
            benchmark_return_pct=bench_annualized_pct,
            alpha_pct=round(port_annualized - bench_annualized_pct, 2),
        )

    def _build_observations(
        self, user: User, v: PortfolioValuation
    ) -> list[HealthObservation]:
        out: list[HealthObservation] = []

        # 1) Concentration — always present.
        out.append(self._concentration_observation(v))

        # 2) Leverage warning if leveraged ETFs at meaningful weight.
        leverage_obs = self._leverage_observation(user, v)
        if leverage_obs is not None:
            out.append(leverage_obs)

        # 3) Cash drag if cash > 25% of total.
        cash_obs = self._cash_drag_observation(user, v)
        if cash_obs is not None:
            out.append(cash_obs)

        # 4) Performance vs benchmark.
        perf_obs = self._performance_observation(user, v)
        if perf_obs is not None:
            out.append(perf_obs)

        # 5) Missing data note.
        if v.missing_prices or v.missing_fx:
            text = []
            if v.missing_prices:
                text.append(
                    f"couldn't price: {', '.join(v.missing_prices)}"
                )
            if v.missing_fx:
                text.append(
                    f"couldn't convert: {', '.join(v.missing_fx)}"
                )
            out.append(HealthObservation(
                severity="warning",
                text=("Some data is missing — " + "; ".join(text) +
                      ". The numbers above exclude those positions."),
            ))

        # Sort by severity (critical > warning > info), keep at most 4.
        sev_order = {"critical": 0, "warning": 1, "info": 2}
        out.sort(key=lambda o: sev_order.get(o.severity, 3))
        return out[:4]

    # -----------------------------------------------------------------
    # Per-observation builders
    # -----------------------------------------------------------------

    def _concentration_observation(self, v: PortfolioValuation) -> HealthObservation:
        c = v.concentration
        flag = c.flag
        if flag == "high":
            top_pos = max(
                (pv for pv in v.valuations if pv.weight_pct is not None),
                key=lambda pv: pv.weight_pct or 0.0,
                default=None,
            )
            if top_pos and top_pos.weight_pct:
                ticker = top_pos.position.ticker
                pct = top_pos.weight_pct
                return HealthObservation(
                    severity="warning",
                    text=(
                        f"{ticker} is {pct:.0f}% of your portfolio — that single "
                        f"position carries most of your outcome. A {self._stress_drawdown_pct(pct):.0f}% "
                        f"drop in {ticker} would take roughly "
                        f"{(pct * 0.30):.0f}% off your total at a 30% drawdown."
                    ),
                )
            return HealthObservation(
                severity="warning",
                text=(
                    f"Top position is {c.top_position_pct:.0f}% of the book and "
                    f"the top 3 are {c.top_3_positions_pct:.0f}% — concentration "
                    "risk is high."
                ),
            )
        if flag == "moderate":
            return HealthObservation(
                severity="info",
                text=(
                    f"Top position is {c.top_position_pct:.0f}% and the top 3 are "
                    f"{c.top_3_positions_pct:.0f}% — concentration is "
                    "moderate but worth tracking as positions move."
                ),
            )
        # low
        return HealthObservation(
            severity="info",
            text=(
                f"Concentration is low — top position is {c.top_position_pct:.0f}% "
                f"and the top 3 are {c.top_3_positions_pct:.0f}%."
            ),
        )

    def _leverage_observation(
        self, user: User, v: PortfolioValuation
    ) -> HealthObservation | None:
        leveraged = [
            pv for pv in v.valuations
            if pv.position.ticker in LEVERAGED_TICKERS
            and pv.weight_pct is not None
            and pv.weight_pct >= 5.0
        ]
        if not leveraged:
            return None
        names = ", ".join(pv.position.ticker for pv in leveraged)
        total_pct = sum(pv.weight_pct or 0 for pv in leveraged)
        # Don't lecture an aggressive trader — note it neutrally.
        severity = "warning" if user.risk_profile != "aggressive" else "info"
        return HealthObservation(
            severity=severity,
            text=(
                f"{names} are leveraged daily-resetting ETFs ({total_pct:.0f}% of "
                "the book combined). They compound differently from their "
                "underlying index over multi-day periods — they're trading "
                "instruments, not buy-and-hold."
            ),
        )

    def _cash_drag_observation(
        self, user: User, v: PortfolioValuation
    ) -> HealthObservation | None:
        if v.total_value <= 0:
            return None
        cash_pct = v.cash_value / v.total_value * 100
        if cash_pct < 25:
            return None
        # For a conservative profile, cash buffer is intentional — note neutrally.
        severity = "info" if user.risk_profile == "conservative" else "info"
        return HealthObservation(
            severity=severity,
            text=(
                f"{cash_pct:.0f}% of the book ({self._fmt_amount(v.cash_value, user.base_currency)}) "
                "is in cash — appropriate as a buffer, but a meaningful drag on "
                "long-run growth if you're not planning to deploy it soon."
            ),
        )

    def _performance_observation(
        self, user: User, v: PortfolioValuation
    ) -> HealthObservation | None:
        if v.total_return_pct is None:
            return None
        bench = self._benchmark(user, v)
        if bench is None or bench.alpha_pct is None:
            return HealthObservation(
                severity="info",
                text=(
                    f"Total return since cost basis is "
                    f"{self._fmt_pct(v.total_return_pct)}."
                ),
            )
        beat = bench.alpha_pct >= 0
        verb = "ahead of" if beat else "behind"
        return HealthObservation(
            severity="info",
            text=(
                f"Approximate annualized return is "
                f"{bench.portfolio_return_pct:+.1f}% — {verb} the {bench.benchmark} "
                f"by {abs(bench.alpha_pct):.1f}% on a similar-period basis. "
                "(Approximation: we don't have your full transaction history.)"
            ),
        )

    # -----------------------------------------------------------------
    # Prose rendering
    # -----------------------------------------------------------------

    def _render_prose(
        self, user: User, report: PortfolioHealthReport
    ) -> Iterable[str]:
        """Yield sentences. Each becomes one stream chunk."""
        yield report.headline

        if report.is_build_oriented:
            for obs in report.observations:
                yield obs.text
            yield (
                "When you're ready to take a first step, ask me to put together "
                "a starter allocation."
            )
            yield report.disclaimer
            return

        for obs in report.observations:
            yield obs.text

        # Closing nudge when concentration is high — actionable but not
        # prescriptive (assignment requires actionable, NOT prescriptive).
        if report.concentration_risk.flag == "high":
            yield (
                "Worth thinking about gradually trimming the top position to "
                "bring it closer to a 25–30% weight, depending on your goals."
            )

        yield report.disclaimer

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _fmt_amount(value: float, currency: str) -> str:
        symbol = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}.get(currency, "")
        return f"{symbol}{value:,.0f}"

    @staticmethod
    def _fmt_pct(value: float) -> str:
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.1f}%"

    @staticmethod
    def _stress_drawdown_pct(weight_pct: float) -> int:
        """How much the position would have to drop for a noticeable hit. Used
        in the high-concentration observation. Just a fixed 30% — kept here
        so the magic number is named."""
        return 30
