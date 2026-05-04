"""Pydantic schemas for the classifier output.

The single LLM call returns a ``ClassificationResult``. The fields are
shaped to match the matcher contract documented in ``fixtures/README.md``:
string-list entities use subset-with-normalization matching; numeric
entities use ±5% tolerance; ``agent`` is exact-match against the
taxonomy.

All defaults are conservative — the schema must validate even when the
LLM returns the minimum useful response. This is what lets the
heuristic fallback emit the same type without scaffolding every field.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Entities(BaseModel):
    """Extracted entities. Empty lists / None mean 'not present', not 'unknown'."""

    tickers: list[str] = Field(
        default_factory=list,
        description="Uppercase ticker symbols. Strip exchange suffix (ASML.AS → ASML).",
    )
    topics: list[str] = Field(
        default_factory=list,
        description="Lowercase short noun phrases (semiconductors, ev, china, rates).",
    )
    sectors: list[str] = Field(
        default_factory=list,
        description="Lowercase sector names if explicitly named.",
    )
    amount: float | None = Field(
        default=None,
        description="Currency amount in base units (10k → 10000, 1m → 1000000).",
    )
    rate: float | None = Field(
        default=None,
        description="Rate as decimal (7% → 0.07, 6.5 → 0.065).",
    )
    period_years: float | None = Field(
        default=None,
        description="Time horizon in years (12 months → 1.0).",
    )


class SafetyVerdict(BaseModel):
    """Informational only — the runtime SafetyGuard is the authority on blocks.

    The classifier may flag a query as suspicious here, but this verdict
    appears only in response metadata. It does NOT change routing.
    """

    is_safe: bool = True
    note: str | None = None


class ClassificationResult(BaseModel):
    """Output of one classifier call."""

    intent: str = Field(
        ...,
        description="Short slug for the user's primary intent (e.g. 'research', 'health_check').",
    )
    agent: str = Field(
        ...,
        description="Exactly one agent name from the taxonomy.",
    )
    entities: Entities = Field(default_factory=Entities)
    confidence: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="1.0 = unambiguous routing; 0.5 = could go multiple ways.",
    )
    reasoning: str = Field(
        "",
        description="One short internal-use sentence; not shown to the end user.",
    )
    safety_verdict: SafetyVerdict = Field(default_factory=SafetyVerdict)
