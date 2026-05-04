"""Market data + FX abstraction.

Two implementations:
  - ``StaticMarketData``  : in-memory dicts. Used by every test, plus the
                             default for runs without a network.
  - ``YFinanceMarketData`` : production. Lazy-imports yfinance, caches
                             results to disk (default 1h TTL for prices,
                             6h for FX) so repeated agent runs don't hit
                             the network again.

Tests must NEVER hit the network. The conftest deletes OPENAI_API_KEY,
and the YFinance class lazily imports yfinance — so a test that
accidentally constructed it without mocking would fail loudly on import,
not silently make a real call.

Both implementations report missing data by returning ``None`` rather
than raising. Metric calculations downstream report missing tickers
explicitly so the user sees "couldn't price X" instead of a blank
portfolio value.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class MarketData(ABC):
    """Synchronous read-only interface to market quotes + FX rates.

    Synchronous because metric computation is CPU-bound and reading
    cached prices is local. Production code can wrap a fetch in
    ``asyncio.to_thread`` if needed.
    """

    @abstractmethod
    def get_price(self, ticker: str) -> float | None:
        """Last known price in the ticker's listed currency."""

    @abstractmethod
    def get_fx_rate(self, from_currency: str, to_currency: str) -> float | None:
        """Multiplier to convert FROM → TO. Returns 1.0 when equal."""


# ---------------------------------------------------------------------------
# Static (test / offline) implementation
# ---------------------------------------------------------------------------

class StaticMarketData(MarketData):
    """In-memory provider. Same currency assumption as yfinance: price for
    a ticker is in the currency of the listing exchange (ASML.AS in EUR,
    7203.T in JPY, etc.)."""

    def __init__(
        self,
        *,
        prices: dict[str, float] | None = None,
        fx_rates: dict[tuple[str, str], float] | None = None,
    ) -> None:
        # Store with case-folded keys so ASML.AS and asml.as both resolve.
        self._prices = {k.upper(): float(v) for k, v in (prices or {}).items()}
        self._fx = {(a.upper(), b.upper()): float(r) for (a, b), r in (fx_rates or {}).items()}

    def get_price(self, ticker: str) -> float | None:
        return self._prices.get(ticker.upper())

    def get_fx_rate(self, from_currency: str, to_currency: str) -> float | None:
        a, b = from_currency.upper(), to_currency.upper()
        if a == b:
            return 1.0
        rate = self._fx.get((a, b))
        if rate is not None:
            return rate
        # Allow inverse lookup so tests only need to register one direction.
        inverse = self._fx.get((b, a))
        if inverse and inverse != 0:
            return 1.0 / inverse
        return None


# ---------------------------------------------------------------------------
# Disk cache used by the production implementation
# ---------------------------------------------------------------------------

class _DiskCache:
    """Trivial JSON-file-per-key cache with a TTL. Kept tiny on purpose:
    real cache invalidation is a runtime concern, this is just a polite
    way to avoid hammering yfinance during a session."""

    def __init__(self, directory: Path, ttl_seconds: float) -> None:
        self._dir = directory
        self._ttl = ttl_seconds

    def _path(self, key: str) -> Path:
        # Replace path separators that show up in FX keys.
        safe = key.replace("/", "_").replace(":", "_")
        return self._dir / f"{safe}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if time.time() - data.get("fetched_at", 0) > self._ttl:
            return None
        return data.get("value")

    def set(self, key: str, value: Any) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._path(key).write_text(
                json.dumps({"fetched_at": time.time(), "value": value})
            )
        except OSError:
            # Cache failure must not break a request.
            return


# ---------------------------------------------------------------------------
# Production: yfinance-backed
# ---------------------------------------------------------------------------

class YFinanceMarketData(MarketData):
    """Live market data via yfinance.

    Lazy-imports yfinance so test runs that exclusively use ``StaticMarketData``
    don't load it.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        price_ttl_s: float = 3600,    # 1h
        fx_ttl_s: float = 6 * 3600,   # 6h — FX moves slowly enough at our resolution
    ) -> None:
        self._price_cache = _DiskCache(
            cache_dir or Path("fixtures/.market_cache/prices"), price_ttl_s
        )
        self._fx_cache = _DiskCache(
            cache_dir or Path("fixtures/.market_cache/fx"), fx_ttl_s
        )

    def get_price(self, ticker: str) -> float | None:
        cached = self._price_cache.get(ticker.upper())
        if cached is not None:
            return float(cached)
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).fast_info
            price = float(info["last_price"])
        except Exception:
            return None
        self._price_cache.set(ticker.upper(), price)
        return price

    def get_fx_rate(self, from_currency: str, to_currency: str) -> float | None:
        a, b = from_currency.upper(), to_currency.upper()
        if a == b:
            return 1.0
        key = f"{a}_{b}"
        cached = self._fx_cache.get(key)
        if cached is not None:
            return float(cached)
        try:
            import yfinance as yf
            symbol = f"{a}{b}=X"
            rate = float(yf.Ticker(symbol).fast_info["last_price"])
        except Exception:
            return None
        self._fx_cache.set(key, rate)
        return rate
