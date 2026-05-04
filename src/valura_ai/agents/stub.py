"""StubAgent — used for every taxonomy agent that isn't implemented yet.

The router's job is to route correctly even when the destination is a
stub. From the assignment:

    > For these, return a structured "not implemented" response that
    > includes: the classified intent, the extracted entities, the agent
    > that would have handled this, a short message indicating the
    > agent is not implemented in this build. Do not crash. Do not
    > return errors.

Each stub instance is constructed with the agent name it stands in for,
so the structured payload reports the correct ``agent`` field.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import ClassVar

from ..classifier.schema import ClassificationResult, Entities
from ..llm.client import ChatMessage
from ..portfolio.models import User
from .base import AgentEvent, BaseAgent
from .schemas import StubResponse


# Per-agent one-line teasers. Keeps the prose specific without sprawling.
# An unknown agent gets a generic but truthful fallback line.
_PURPOSE: dict[str, str] = {
    "market_research":      "fundamentals, recent moves, and competitive context for a specific company or sector",
    "investment_strategy":  "allocation, rebalancing, and named-strategy decisions",
    "financial_calculator": "future-value, retirement, and amortization projections",
    "risk_assessment":      "VaR, drawdown, stress tests, and exposure analytics",
    "recommendations":      "buy/sell/add suggestions grounded in your holdings",
    "predictive_analysis":  "rate, FX, commodity, and earnings-season forecasts",
    "support":              "platform and account questions (statements, tax docs, password)",
}


class StubAgent(BaseAgent):
    """Streaming stub for an unimplemented specialist agent."""

    name: ClassVar[str] = "_stub"  # placeholder; instances stand in for a specific agent

    def __init__(self, *, agent_name: str) -> None:
        if not agent_name:
            raise ValueError("StubAgent requires a non-empty agent_name")
        self._agent_name = agent_name

    async def stream(
        self,
        *,
        classification: ClassificationResult,
        user: User | None = None,
        history: Sequence[ChatMessage] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        purpose = _PURPOSE.get(
            self._agent_name,
            "this specialist's responsibilities",
        )

        # Mention the captured entities so the user sees their request
        # was understood, not lost.
        captured = self._summarise_entities(classification.entities)
        captured_clause = (
            f" I captured: {captured}." if captured else ""
        )

        message = (
            f"The {self._agent_name} specialist isn't implemented in this build "
            f"— that agent handles {purpose}.{captured_clause} Routing this "
            "correctly today; the specialist will land in an upcoming release."
        )

        yield AgentEvent(type="token", text=message)
        yield AgentEvent(
            type="structured",
            data=StubResponse(
                intent=classification.intent,
                agent=self._agent_name,
                entities=classification.entities,
                message=message,
            ),
        )

    @staticmethod
    def _summarise_entities(entities: Entities) -> str:
        parts: list[str] = []
        if entities.tickers:
            parts.append(f"tickers={entities.tickers}")
        if entities.topics:
            parts.append(f"topics={entities.topics}")
        if entities.sectors:
            parts.append(f"sectors={entities.sectors}")
        if entities.amount is not None:
            parts.append(f"amount={entities.amount}")
        if entities.rate is not None:
            parts.append(f"rate={entities.rate}")
        if entities.period_years is not None:
            parts.append(f"period_years={entities.period_years}")
        return ", ".join(parts)
