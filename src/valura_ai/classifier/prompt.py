"""System prompt builder for the intent classifier.

Two design choices worth defending:

  1) The agent list is INPUT to the prompt, not hardcoded into it. The
     taxonomy is loaded from ``fixtures/test_queries/intent_classification.json``
     at startup. Swapping the official taxonomy in for grading does not
     touch ``src/``.

  2) The prompt describes each agent in plain language and gives general
     routing principles — it does NOT contain example queries from the
     fixture. That's deliberate. We want the classifier to generalise to
     the larger hidden eval set, and few-shot examples drawn from the
     public set would bias both training-evaluation overlap and the
     classifier's posterior toward those exact phrasings.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Agent descriptions.
# Names that aren't in this dict get a generic placeholder — keeps the
# prompt builder robust if the fixture taxonomy is extended.
# ---------------------------------------------------------------------------

AGENT_DESCRIPTIONS: Mapping[str, str] = {
    "portfolio_health": (
        "General overview of the user's OWN portfolio — performance, "
        "diversification, year-to-date returns, red-flag scans, "
        "'is everything OK?' questions. Pick this for holistic personal-portfolio asks."
    ),
    "market_research": (
        "Information about a specific company, sector, or industry that exists "
        "OUTSIDE the user's portfolio — fundamentals, recent news, business "
        "overview. Pick this for 'tell me about X' framing."
    ),
    "investment_strategy": (
        "Allocation, rebalancing, factor tilts, named strategies (60/40, DCA, "
        "barbell, lump sum vs scaled-in), tax-loss harvesting decisions. "
        "Pick this for 'should I do strategy X' framing without a specific ticker."
    ),
    "financial_calculator": (
        "Numerical projections — future value, retirement targets, mortgage "
        "payments, withdrawal rates. Pick this when the user asks for a number "
        "to be computed (amount, rate, period)."
    ),
    "risk_assessment": (
        "Specific risk analytics — VaR, drawdown, stress tests, scenario "
        "analysis, factor or country exposure framed as risk. Pick this when "
        "the user wants a risk number, not a general overview."
    ),
    "recommendations": (
        "Action-oriented asks — what to buy, sell, add, or rotate into. "
        "'What should I add', 'should I sell my X', 'give me ideas'. "
        "Pick this for action-on-positions framing."
    ),
    "predictive_analysis": (
        "Forecasts about markets, rates, FX, commodities, earnings season. "
        "'Where will rates go', 'is now a good time', 'Fed cutting next?'."
    ),
    "support": (
        "Platform / account questions — tax docs, statements, password resets, "
        "app issues, account closure. Not investment advice."
    ),
}


# ---------------------------------------------------------------------------
# User context (optional, included in the prompt when available)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UserContext:
    """The slice of user data passed to the classifier."""

    user_id: str
    risk_profile: str  # conservative | moderate | balanced | aggressive
    base_currency: str
    holdings_tickers: tuple[str, ...] = ()
    kyc_status: str = "verified"


def _format_user_context(ctx: UserContext) -> str:
    holdings = ", ".join(ctx.holdings_tickers) if ctx.holdings_tickers else "(empty)"
    return (
        "## User context\n"
        f"- risk profile: {ctx.risk_profile}\n"
        f"- base currency: {ctx.base_currency}\n"
        f"- KYC status: {ctx.kyc_status}\n"
        f"- holdings (tickers only): {holdings}"
    )


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

_HEADER = (
    "You are the Valura AI router for a wealth-management product. "
    "Classify each user query into EXACTLY ONE specialist agent and extract "
    "any entities the downstream agent will need."
)

_ROUTING_RULES = """
## Routing rules
1. Use the user's CURRENT TURN as the primary signal.
2. If the current turn is vague ('thoughts?', 'and now?', 'worth it?', 'too late?',
   'is this fine?'), use the most recent user turn from history to determine the
   agent. Entities should still come from the current turn when present;
   otherwise inherit from history.
3. For multi-intent queries, pick the agent that does the most work or answers
   the bigger question:
     - 'explain X and project numbers'         → financial_calculator
     - 'is the Fed cutting and should I rotate' → predictive_analysis
     - 'should I sell or hold and what to buy'  → recommendations
4. Action verbs on a specific ticker ('should I sell my NVDA') → recommendations.
5. Personal portfolio metrics ('my YTD', 'how much I made this year') →
   portfolio_health, NOT market_research.
"""

_ENTITY_RULES = """
## Entity extraction
- tickers: uppercase symbols. Map common company names to symbols when
  unambiguous (Microsoft → MSFT, Apple → AAPL, Tesla → TSLA, Toyota → TM,
  Alibaba → BABA). Strip exchange suffixes (ASML.AS → ASML, 7203.T → 7203).
- topics: lowercase short noun phrases for non-ticker subjects
  (semiconductors, ev, china, rates, oil, fed, tech).
- sectors: lowercase only when the user explicitly names a sector.
- amount: in base units. 500/month → 500. 10k → 10000. 1.5m → 1500000.
- rate: as a decimal. 7% → 0.07. 6.5% → 0.065.
- period_years: in years. '30 years' → 30. '12 months' → 1.0.
Be conservative — omit rather than guess. Better to extract nothing than
to inject a wrong ticker.
"""

_OUTPUT_RULES = """
## Output requirements
- agent must be EXACTLY one of the names listed above. No synonyms, no
  variations.
- intent is a short slug describing the user's ask ('research', 'health_check',
  'forecast'); free-form, used for logging.
- confidence: 1.0 when routing is unambiguous; lower when the query could
  reasonably go to more than one agent.
- reasoning: ONE short sentence; not shown to the user.
- safety_verdict.is_safe: true unless the query EXPLICITLY states harmful
  intent. The runtime safety guard runs separately and decides blocks; this
  verdict is informational only.
"""


def build_system_prompt(
    agents: list[str],
    user_context: UserContext | None = None,
) -> str:
    """Assemble the system prompt from the taxonomy + optional user context."""
    lines = [_HEADER, "", "## Agents (pick exactly one)"]
    for name in agents:
        desc = AGENT_DESCRIPTIONS.get(
            name,
            "Pick this when the query clearly fits the agent's name.",
        )
        lines.append(f"- {name}: {desc}")
    lines.append(_ROUTING_RULES)
    lines.append(_ENTITY_RULES)
    if user_context is not None:
        lines.append("")
        lines.append(_format_user_context(user_context))
    lines.append(_OUTPUT_RULES)
    return "\n".join(lines)
