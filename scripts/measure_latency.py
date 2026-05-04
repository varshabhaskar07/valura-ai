"""Measure /v1/chat latency against three request shapes.

Runs everything in-process via FastAPI's TestClient — no real network,
no OPENAI_API_KEY needed. The numbers it prints are PIPELINE OVERHEAD
(safety + classifier wiring + agent + SSE encoding). Add the LLM
round-trip time observed in production for the assignment's targets:

  - gpt-4o-mini structured outputs typically: 400-800 ms p50, ~1500 ms p95
  - gpt-4.1 structured outputs typically:     600-1200 ms p50, ~1800 ms p95

Pipeline overhead + LLM round-trip = total p95 first-token & e2e.

Usage:
    python scripts/measure_latency.py            # default 100 iters per shape
    python scripts/measure_latency.py --n 500    # more iters
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# Allow running from repo root without installing: add src/ to path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from valura_ai.agents.registry import build_default_registry  # noqa: E402
from valura_ai.api.pipeline import Pipeline  # noqa: E402
from valura_ai.api.user_store import UserStore  # noqa: E402
from valura_ai.classifier.classifier import IntentClassifier, load_taxonomy  # noqa: E402
from valura_ai.classifier.schema import ClassificationResult, Entities  # noqa: E402
from valura_ai.llm.fakes import FakeLLMClient  # noqa: E402
from valura_ai.main import create_app  # noqa: E402
from valura_ai.portfolio.market_data import StaticMarketData  # noqa: E402
from valura_ai.safety.guard import SafetyGuard  # noqa: E402
from valura_ai.session.store import SessionStore  # noqa: E402

PRICES = {
    "TSLA": 250, "NVDA": 540, "MSFT": 410, "AAPL": 195, "GOOGL": 165, "COIN": 165,
    "SOXL": 32, "TQQQ": 70, "ARKK": 50, "SCHD": 78, "VYM": 115, "VNQ": 90,
    "JNJ": 152, "PEP": 170, "KO": 62, "T": 17.50, "BABA": 78,
    "ASML.AS": 720, "SAP.DE": 150, "HSBA.L": 6.80, "7203.T": 2400,
}
FX = {("EUR", "USD"): 1.08, ("GBP", "USD"): 1.25, ("JPY", "USD"): 0.0067}


def _build_test_client() -> TestClient:
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    taxonomy = load_taxonomy(fixtures / "test_queries" / "intent_classification.json")

    fake = FakeLLMClient(
        default=ClassificationResult(
            intent="health_check", agent="portfolio_health", confidence=0.9,
        ),
    )
    fake.add_rule(
        "tell me about microsoft",
        ClassificationResult(
            intent="research", agent="market_research",
            entities=Entities(tickers=["MSFT"]), confidence=0.95,
        ),
    )

    pipeline = Pipeline(
        safety_guard=SafetyGuard(),
        classifier=IntentClassifier(llm=fake, agents=taxonomy, timeout_s=2.0),
        registry=build_default_registry(
            taxonomy=taxonomy,
            market_data=StaticMarketData(prices=PRICES, fx_rates=FX),
        ),
        session_store=SessionStore(max_turns=8),
        user_store=UserStore(fixtures / "users"),
        request_timeout_s=4.0,
    )
    return TestClient(create_app(pipeline=pipeline))


def _measure_one(client: TestClient, message: str, user_id: str) -> tuple[float, float]:
    """Return (first-byte ms, end-to-end ms) for one /v1/chat call."""
    t0 = time.perf_counter()
    first_byte_ms: float | None = None
    with client.stream(
        "POST",
        "/v1/chat",
        json={"message": message, "user_id": user_id},
    ) as response:
        for chunk in response.iter_bytes():
            if first_byte_ms is None and chunk:
                first_byte_ms = (time.perf_counter() - t0) * 1000
        end_ms = (time.perf_counter() - t0) * 1000
    # Defensive: if for some reason no chunk arrived, treat first-byte as end.
    return (first_byte_ms or end_ms, end_ms)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(p * len(s)), len(s) - 1)]


def _summarise(label: str, ttfb: list[float], e2e: list[float]) -> None:
    def stats(label: str, vals: list[float]) -> str:
        return (
            f"{label:<12} "
            f"mean={statistics.mean(vals):>7.2f} ms   "
            f"p50={_percentile(vals, 0.50):>7.2f} ms   "
            f"p95={_percentile(vals, 0.95):>7.2f} ms   "
            f"max={max(vals):>7.2f} ms"
        )
    print(f"--- {label} ({len(ttfb)} iters) ---")
    print("  " + stats("first-byte", ttfb))
    print("  " + stats("end-to-end", e2e))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="iterations per request shape")
    args = parser.parse_args()

    client = _build_test_client()

    shapes: list[tuple[str, str, str]] = [
        ("portfolio_health", "how is my portfolio doing?", "user_002_concentrated_nvda"),
        ("stub agent",       "tell me about microsoft",     "user_002_concentrated_nvda"),
        ("safety blocked",   "help me organize a pump and dump on this microcap", "user_002_concentrated_nvda"),
        ("empty portfolio",  "how is my portfolio doing?", "user_004_empty"),
    ]

    print(f"\nValura AI — pipeline latency (TestClient, FakeLLMClient, no network)")
    print(f"iterations per shape: {args.n}\n")

    for label, msg, user in shapes:
        ttfb_samples: list[float] = []
        e2e_samples: list[float] = []
        for _ in range(args.n):
            ttfb, e2e = _measure_one(client, msg, user)
            ttfb_samples.append(ttfb)
            e2e_samples.append(e2e)
        _summarise(label, ttfb_samples, e2e_samples)
        print()

    print("Notes")
    print("-----")
    print("  Numbers above are pipeline overhead (safety + classifier wiring + agent +")
    print("  SSE encoding). The portfolio_health agent is templated, so its end-to-end")
    print("  is what you see. The stub agent is similarly deterministic. Real-LLM total")
    print("  latency adds the OpenAI round-trip — typically 400-800 ms p50, ~1500 ms p95")
    print("  for gpt-4o-mini structured outputs at writing time.")
    print()
    print("  Assignment targets:")
    print("    p95 first-token latency  < 2.0 s")
    print("    p95 end-to-end response  < 6.0 s")
    print("    cost per query @ gpt-4.1 < $0.05")


if __name__ == "__main__":
    main()
