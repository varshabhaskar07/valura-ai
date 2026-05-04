"""AgentRegistry — name → BaseAgent lookup.

Built once per process and shared across requests. Adding a new
specialist later is a one-liner at construction time:

    registry = AgentRegistry(
        taxonomy=load_taxonomy(),
        implemented={
            "portfolio_health":   PortfolioHealthAgent(market_data=md),
            "financial_calculator": FinancialCalculatorAgent(...),  # future
        },
    )

Every taxonomy name resolves to *something*: implemented agent if present,
``StubAgent`` otherwise. Unknown names (which shouldn't happen if the
classifier obeys its taxonomy) get a fresh stub rather than raising —
the assignment requires the router not to crash on routing.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..portfolio.market_data import MarketData
from .base import BaseAgent
from .portfolio_health import PortfolioHealthAgent
from .stub import StubAgent


class AgentRegistry:
    """Maps every taxonomy agent name to a BaseAgent instance."""

    __slots__ = ("_agents", "_implemented_names")

    def __init__(
        self,
        *,
        taxonomy: list[str],
        implemented: Mapping[str, BaseAgent] | None = None,
    ) -> None:
        if not taxonomy:
            raise ValueError("taxonomy must be non-empty")
        impl = dict(implemented or {})
        # Reject implemented agents that aren't in the taxonomy — that
        # would silently never be reached, which is almost always a bug.
        unknown_impl = [name for name in impl if name not in taxonomy]
        if unknown_impl:
            raise ValueError(
                f"implemented agents not in taxonomy: {unknown_impl}. "
                "Add them to the taxonomy first."
            )
        self._agents: dict[str, BaseAgent] = {
            name: impl.get(name) or StubAgent(agent_name=name)
            for name in taxonomy
        }
        self._implemented_names: frozenset[str] = frozenset(impl.keys())

    def get(self, name: str) -> BaseAgent:
        """Return the agent for ``name``. Falls back to a fresh stub for
        unknown names so a misclassified request still returns a valid
        structured response instead of crashing."""
        agent = self._agents.get(name)
        if agent is not None:
            return agent
        return StubAgent(agent_name=name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._agents)

    @property
    def implemented(self) -> frozenset[str]:
        return self._implemented_names

    def is_implemented(self, name: str) -> bool:
        return name in self._implemented_names


def build_default_registry(
    *,
    taxonomy: list[str],
    market_data: MarketData,
) -> AgentRegistry:
    """Production wiring: portfolio_health implemented, every other taxonomy
    name stubbed. Future steps add more entries to ``implemented``.
    """
    return AgentRegistry(
        taxonomy=taxonomy,
        implemented={
            "portfolio_health": PortfolioHealthAgent(market_data=market_data),
        },
    )
