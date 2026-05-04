"""Deterministic, keyword + regex fallback classifier.

Runs when the LLM call raises ANY ``LLMError`` (timeout, parse, refusal,
auth). Pure Python, no I/O. Returns the same ``ClassificationResult``
schema as the primary LLM path so downstream code can't tell which
branch produced the output.

Design tradeoff: this is a SAFETY NET, not a competitive classifier. It
will mis-route some queries that the LLM would handle correctly. We
accept that — the alternative is crashing or returning a wrong-typed
response, both of which are worse for the request lifecycle.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .schema import ClassificationResult, Entities, SafetyVerdict

if TYPE_CHECKING:
    from ..llm.client import ChatMessage


# ---------------------------------------------------------------------------
# Keyword sets per agent
# Each tuple is a set of phrases that, if present in the query, score 1
# point toward that agent. Agent with the highest score wins; ties broken
# in registration order.
# ---------------------------------------------------------------------------

AGENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "support": (
        "tax docs", "1099", "password", "app down", "app crash", "is the app",
        "close my account", "close the account", "export my", "where is my 1099",
        "how do i export", "statement",
    ),
    "financial_calculator": (
        "future value", "compound interest", "compound", "retire at",
        "retire by", "retirement target", "loan payment", "mortgage",
        "monthly contribution", "withdrawal rate", "swr", "amortis",
        "/month at", "/mo at", "a month at", "per month at",
        "what do i end up", "what does", "become in",
        "if i put", "if i invest", "if i save",
    ),
    "predictive_analysis": (
        "outlook", "forecast", "prediction", "where will", "where do you think",
        "where is the", "by year end", "year end", "next meeting",
        "fed cutting", "fed hiking", "fed will", "rate cut", "rate hike",
        "is now a good time", "too late", "what about now",
    ),
    "risk_assessment": (
        "var ", "value at risk", "downside", "drawdown", "stress test",
        "scenario", "exposed to", "exposure to", "sharpe", "spike",
        "how worried", "worried about", "risking with", "concentration risk",
    ),
    "investment_strategy": (
        "rotate", "60/40", "60 40", "international exposure", "small cap",
        "barbell", "dollar cost average", "dollar cost", " dca ", "lump sum",
        "tax loss harvest", "tax-loss harvest", "rebalance", "rebal ",
        "drift threshold", "all in or", "rebal now",
        "split it", "tilt small", "tilt international",
    ),
    "recommendations": (
        "what should i", "what stocks should", "what etfs should",
        "should i sell", "should i buy", "should i add", "should i hold",
        "any ideas", "give me a few", "give me some", "give me a rec",
        "good etfs for", "round out", "where should i put",
        "what to buy", "what would u do", "what would you do",
        "tell me good stocks", "good stocks",
        "moar", "more like that", "more recs",
    ),
    "market_research": (
        "tell me about", "what's the latest on", "whats the latest",
        "what does", "fundamentals", "background on", "info on the",
        "info on", "what's going on", "whats going on", "happening with",
        "wtf is", "bull case", "bear case", "deep dive",
    ),
    "portfolio_health": (
        "my portfolio", "portfolio doing", "portfolio's healthy",
        "portfolio healthy", "portfolio in good shape", "diversif",
        "health check", "any red flags", "red flags", "rundown",
        "investments are looking", "portfolio overview", "ytd",
        "year to date", "year-to-date", "how much have i made",
        "how much I made", "money safe", "is my money safe",
        "is this fine",
    ),
}


# Topic detection — independent of agent. A heuristic enrichment, not a
# definitive set; the matcher only requires subset overlap so omitting a
# topic loses one row, never the whole eval.
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "semiconductor": ("semiconductor", "semis", "chip"),
    "ev":            ("ev stock", "ev stocks", " ev ", "electric vehicle"),
    "china":         ("china", "chinese"),
    "rates":         ("rates", "interest rate", "rate spike", "rates spike"),
    "fed":           ("fed ", "federal reserve", "fomc"),
    "oil":           ("oil ", "crude"),
    "growth":        ("growth to", "growth stock"),
    "value":         ("to value", "value stock", "value tilt"),
    "60/40":         ("60/40", "60 40"),
    "international": ("international",),
    "small cap":     ("small cap", "small-cap"),
    "barbell":       ("barbell",),
    "dca":           (" dca ", "dollar cost"),
    "lump sum":      ("lump sum", "all in or"),
    "tax loss harvesting": ("tax loss harvest", "tax-loss harvest", "tlh"),
    "rebalance":     ("rebalance", "rebal "),
    "var":           ("var ", "value at risk"),
    "drawdown":      ("drawdown",),
    "stress test":   ("stress test",),
    "2008":          ("2008",),
    "tech":          ("tech selloff", " tech ", "tech earnings"),
    "sharpe":        ("sharpe",),
    "options":       ("options ", "options position"),
    "swr":           ("swr", "withdrawal rate"),
    "etf":           ("etf",),
    "s&p":           ("s&p", "s and p"),
    "usd":           ("usd",),
    "eur":           ("eur ",),
    "earnings":      ("earnings",),
    "tax":           (" tax ", "tax docs"),
    "1099":          ("1099",),
    "password":      ("password",),
    "app":           ("the app", "app down"),
    "account":       ("my account", "account",),
    "bonds":         ("bonds", "bond "),
}


# Common company → ticker mappings. Only the very-unambiguous ones —
# "ford" mapping to F is fine; "apple" to AAPL is fine; we do NOT map
# generic words like "amazon" if the query is about the rainforest, but
# the assignment context ensures finance interpretation.
NAME_TO_TICKER: dict[str, str] = {
    "microsoft": "MSFT",
    "apple": "AAPL",
    "tesla": "TSLA",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "nvidia": "NVDA",
    "intel": "INTC",
    "amd": "AMD",
    "palantir": "PLTR",
    "asml": "ASML",
    "jpmorgan": "JPM",
    "j.p. morgan": "JPM",
    "j.p.morgan": "JPM",
    "toyota": "TM",
    "alibaba": "BABA",
    "sap": "SAP",
    "hsbc": "HSBC",
    "pfizer": "PFE",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
}


# Tokens that look like uppercase tickers but aren't — strip them out so
# regex extraction doesn't pollute the entity list.
NON_TICKER_TOKENS: frozenset[str] = frozenset({
    "I", "A", "AI", "AML", "BSA", "MNPI", "DCA", "EV", "ETF", "REIT",
    "IPO", "FX", "SEC", "CEO", "CFO", "CTO", "USD", "EUR", "GBP", "JPY",
    "USA", "US", "UK", "EU", "VAR", "P", "E", "PE", "M&A", "K", "M", "B",
    "T", "OK", "OR", "AND", "BUT", "YTD", "TLH", "0DTE", "HELOC", "VAR",
    "MR", "DR", "CEO", "CFO", "ASAP", "FAQ", "DIY", "TBD", "NA", "TBA",
})


# ---------------------------------------------------------------------------
# Regexes for entity extraction
# ---------------------------------------------------------------------------

_TICKER_REGEX = re.compile(r"(?:\$)?\b([A-Z]{1,5}(?:\.[A-Z]{1,3})?)\b")
_AMOUNT_REGEX = re.compile(
    r"\$?\b(\d+(?:\.\d+)?)\s*(k|m|b|million|billion|thousand)?\b",
    re.IGNORECASE,
)
_RATE_REGEX = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:%|percent|pct)\b",
    re.IGNORECASE,
)
# Bare integer + 'at' + (percent number) pattern: "compound interest on 10k for 20yr at 6"
_BARE_RATE_AT_REGEX = re.compile(r"\bat\s+(\d+(?:\.\d+)?)\s*(?!\w)")
_PERIOD_REGEX = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:yr|years?|y)\b",
    re.IGNORECASE,
)
_PERIOD_MONTHS_REGEX = re.compile(r"\b(\d+(?:\.\d+)?)\s*months?\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Heuristic classifier
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeuristicClassifier:
    """Stateless deterministic classifier."""

    agents: tuple[str, ...]
    default_agent: str = "portfolio_health"

    def classify(
        self,
        query: str,
        history: Sequence["ChatMessage"] | None = None,
    ) -> ClassificationResult:
        agent = self._route(query, history)
        entities = self._extract(query)
        return ClassificationResult(
            intent=agent,
            agent=agent,
            entities=entities,
            confidence=0.4,  # heuristic is never highly confident
            reasoning="heuristic-fallback: keyword routing on current turn"
                      + ("" if not history else " (history considered)"),
            safety_verdict=SafetyVerdict(is_safe=True),
        )

    # ---- routing -----------------------------------------------------------

    def _route(
        self,
        query: str,
        history: Sequence["ChatMessage"] | None,
    ) -> str:
        agent = self._route_one(query)
        if agent is not None:
            return agent
        # Fall back to history's most recent user turn if the current query
        # had no signal — handles "thoughts?", "and now?", etc.
        if history:
            for msg in reversed(history):
                if msg.role == "user":
                    parent = self._route_one(msg.content)
                    if parent is not None:
                        return parent
                    break
        return self.default_agent

    def _route_one(self, text: str) -> str | None:
        text_lc = " " + text.lower().strip() + " "
        # Numerical patterns are an unambiguous calculator signal.
        if (
            _RATE_REGEX.search(text)
            and (_PERIOD_REGEX.search(text) or _PERIOD_MONTHS_REGEX.search(text)
                 or "month" in text_lc or "year" in text_lc)
        ):
            return "financial_calculator"

        scores: dict[str, int] = {}
        for agent, kws in AGENT_KEYWORDS.items():
            if agent not in self.agents:
                continue
            score = sum(1 for kw in kws if kw in text_lc)
            if score:
                scores[agent] = score
        if not scores:
            return None
        # Highest score wins; ties broken by AGENT_KEYWORDS dict order.
        # That order is intentionally support → calculator → predictive →
        # risk → strategy → recommendations → research → health, putting
        # narrower agents first so they aren't drowned out by the broader
        # ones.
        best = max(scores.values())
        for agent in AGENT_KEYWORDS:
            if scores.get(agent) == best and agent in self.agents:
                return agent
        return None  # pragma: no cover — defensive

    # ---- entity extraction ------------------------------------------------

    def _extract(self, query: str) -> Entities:
        text_lc = query.lower()

        tickers: list[str] = []
        # Symbol regex on the original (preserves case).
        for raw in _TICKER_REGEX.findall(query):
            symbol = raw.split(".")[0].upper()
            if symbol in NON_TICKER_TOKENS or len(symbol) < 2:
                continue
            if symbol not in tickers:
                tickers.append(symbol)
        # Company-name lookup on the lowercased text.
        for name, sym in NAME_TO_TICKER.items():
            if name in text_lc and sym not in tickers:
                tickers.append(sym)

        topics: list[str] = []
        for topic, kws in TOPIC_KEYWORDS.items():
            if any(kw in (" " + text_lc + " ") for kw in kws):
                topics.append(topic)

        amount = self._extract_amount(query)
        rate = self._extract_rate(query)
        period = self._extract_period(query)

        return Entities(
            tickers=tickers,
            topics=topics,
            sectors=[],
            amount=amount,
            rate=rate,
            period_years=period,
        )

    @staticmethod
    def _extract_amount(query: str) -> float | None:
        # Find the first number with k/m/b suffix or with a $; otherwise the
        # first plain number that LOOKS like an amount (>= 100). Skips
        # numbers immediately followed by '%' or 'years' (those are rate or
        # period).
        skip_spans: list[tuple[int, int]] = []
        for m in _RATE_REGEX.finditer(query):
            skip_spans.append(m.span())
        for m in _PERIOD_REGEX.finditer(query):
            skip_spans.append(m.span())
        for m in _PERIOD_MONTHS_REGEX.finditer(query):
            skip_spans.append(m.span())

        for m in _AMOUNT_REGEX.finditer(query):
            if any(start <= m.start() < end for start, end in skip_spans):
                continue
            n_str, suffix = m.group(1), (m.group(2) or "").lower()
            n = float(n_str)
            if suffix in ("k", "thousand"):
                return n * 1_000
            if suffix in ("m", "million"):
                return n * 1_000_000
            if suffix in ("b", "billion"):
                return n * 1_000_000_000
            # Treat plain integers >= 100 with $ prefix or context as amounts.
            if "$" in query[max(0, m.start() - 1): m.end() + 1]:
                return n
            if n >= 100:
                return n
        return None

    @staticmethod
    def _extract_rate(query: str) -> float | None:
        m = _RATE_REGEX.search(query)
        if m:
            return round(float(m.group(1)) / 100.0, 6)
        # Pattern "at 6" or "at 7" without explicit %.
        m = _BARE_RATE_AT_REGEX.search(query)
        if m:
            n = float(m.group(1))
            if 0 < n <= 30:  # plausible interest rate
                return round(n / 100.0, 6)
        return None

    @staticmethod
    def _extract_period(query: str) -> float | None:
        m = _PERIOD_REGEX.search(query)
        if m:
            return float(m.group(1))
        m = _PERIOD_MONTHS_REGEX.search(query)
        if m:
            return float(m.group(1)) / 12.0
        return None
