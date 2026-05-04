"""Structured-output schemas for the implemented agents and the stub.

The PortfolioHealthReport shape matches the assignment's reference
output (concentration_risk, performance, benchmark_comparison,
observations, disclaimer) plus a couple of additions:

  - ``user_id`` so logged reports are joinable to the user
  - ``headline`` for a one-line summary the UI can render prominently
  - ``is_build_oriented`` so the consumer can render an empty-portfolio
    response with a different layout (no metrics charts)
  - ``missing_data`` so the consumer can show which tickers/currencies
    couldn't be priced rather than silently dropping them
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..classifier.schema import Entities


class HealthObservation(BaseModel):
    severity: Literal["info", "warning", "critical"]
    text: str = Field(..., min_length=1)


class ConcentrationRisk(BaseModel):
    top_position_pct: float | None = None
    top_3_positions_pct: float | None = None
    flag: Literal["low", "moderate", "high", "n/a"] = "n/a"


class PerformanceSummary(BaseModel):
    total_return_pct: float | None = None
    annualized_return_pct: float | None = None


class BenchmarkComparison(BaseModel):
    benchmark: str
    portfolio_return_pct: float | None = None
    benchmark_return_pct: float | None = None
    alpha_pct: float | None = None


class MissingData(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    currencies: list[str] = Field(default_factory=list)


class StubResponse(BaseModel):
    """Structured payload from the stub agent.

    Preserves the classifier's view of the request — intent, target agent,
    extracted entities — so a downstream consumer can render a sensible
    placeholder UI ("we noted you asked about NVDA + market_research; that
    specialist is not yet available") and so the request is logged with
    enough context to evaluate later.
    """

    intent: str
    agent: str
    entities: Entities
    status: Literal["not_implemented"] = "not_implemented"
    message: str


class PortfolioHealthReport(BaseModel):
    user_id: str
    headline: str = Field(..., min_length=1)
    concentration_risk: ConcentrationRisk = Field(default_factory=ConcentrationRisk)
    performance: PerformanceSummary = Field(default_factory=PerformanceSummary)
    benchmark_comparison: BenchmarkComparison | None = None
    observations: list[HealthObservation] = Field(default_factory=list)
    missing_data: MissingData = Field(default_factory=MissingData)
    is_build_oriented: bool = False
    disclaimer: str = Field(..., min_length=1)
