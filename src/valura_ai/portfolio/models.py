"""Portfolio domain models.

Pydantic-typed so they validate at the boundary (loading from fixture JSON,
or arriving in API requests). Mutating ``Portfolio`` is allowed (positions
move over time); ``Position`` itself is frozen because individual rows
should be replaced wholesale rather than mutated in place.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Position(BaseModel):
    """A single holding."""

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(..., min_length=1, description="Symbol; may include exchange suffix.")
    exchange: str = Field("", description="Free-form exchange tag (NASDAQ, LSE, XETRA, ...).")
    quantity: float = Field(..., gt=0)
    cost_basis_per_share: float = Field(..., gt=0)
    currency: str = Field(..., min_length=3, max_length=3, description="ISO 4217.")

    @field_validator("currency")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @property
    def cost_basis(self) -> float:
        """Total cost basis in this position's currency."""
        return self.quantity * self.cost_basis_per_share


class KYC(BaseModel):
    status: str
    country: str = ""


class Portfolio(BaseModel):
    """Holdings + cash. Cash is per-currency; convert at use site, not load site."""

    positions: list[Position] = Field(default_factory=list)
    cash: dict[str, float] = Field(default_factory=dict)

    @field_validator("cash")
    @classmethod
    def _upper_keys(cls, v: dict[str, float]) -> dict[str, float]:
        return {k.upper(): float(amt) for k, amt in v.items()}

    @property
    def is_empty(self) -> bool:
        """True iff the user has no positions. Cash alone is still 'empty' for
        portfolio-health purposes (handled in the agent's BUILD branch)."""
        return not self.positions

    @property
    def total_cost_basis_per_currency(self) -> dict[str, float]:
        """Cost basis grouped by currency. Useful for return calculations
        before FX conversion to the base."""
        out: dict[str, float] = {}
        for p in self.positions:
            out[p.currency] = out.get(p.currency, 0.0) + p.cost_basis
        return out


class User(BaseModel):
    """One user as represented in fixtures/users/*.json."""

    user_id: str
    display_name: str = ""
    kyc: KYC = Field(default_factory=lambda: KYC(status="verified"))
    risk_profile: str = "moderate"
    base_currency: str = Field("USD", min_length=3, max_length=3)
    portfolio: Portfolio = Field(default_factory=Portfolio)
    notes: str = ""

    @field_validator("base_currency")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @classmethod
    def from_fixture(cls, source: str | bytes | Path | dict) -> "User":
        """Load a user from a fixture file path or a parsed dict."""
        if isinstance(source, Path):
            return cls.model_validate_json(source.read_text())
        if isinstance(source, (bytes, str)):
            try:
                # Try as raw JSON text first
                return cls.model_validate_json(source)
            except Exception:
                pass
            return cls.model_validate(json.loads(source))
        return cls.model_validate(source)
