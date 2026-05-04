"""FastAPI app factory + production wiring.

Run with:
    uvicorn valura_ai.main:app --reload

Env-driven (see .env.example):
  - OPENAI_API_KEY        — when set, uses the real OpenAI client.
                            When unset, falls back to a stub LLM that
                            classifies everything as portfolio_health, so
                            the demo at least returns valid responses
                            without a key.
  - VALURA_MODEL          — model id (gpt-4o-mini in dev, gpt-4.1 in eval)
  - VALURA_REQUEST_TIMEOUT_S
  - VALURA_CLASSIFIER_TIMEOUT_S
  - VALURA_HISTORY_TURNS
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI

from .agents.registry import build_default_registry
from .api.pipeline import Pipeline
from .api.routes import router as v1_router
from .api.user_store import UserStore
from .classifier.classifier import IntentClassifier, load_taxonomy
from .classifier.schema import ClassificationResult
from .config import Settings, get_settings
from .llm.client import LLMClient, OpenAILLMClient
from .llm.fakes import FakeLLMClient
from .portfolio.market_data import MarketData, YFinanceMarketData
from .safety.guard import SafetyGuard
from .session.store import SessionStore

logger = logging.getLogger(__name__)


def _build_llm_client(settings: Settings) -> LLMClient:
    """Real OpenAI client if a key is set; otherwise a stub fake.

    Why fall back to a fake instead of failing: the assignment requires
    `pip install` + `uvicorn` to bring the service up locally. A
    candidate or reviewer running without an API key should still see
    the safety + routing pipeline work end-to-end. The fake routes
    everything to portfolio_health (the implemented agent) — the
    response is real even if the classification is degraded.
    """
    if settings.openai_api_key:
        return OpenAILLMClient(
            api_key=settings.openai_api_key,
            model=settings.model,
            default_timeout_s=settings.classifier_timeout_s,
        )
    logger.warning(
        "OPENAI_API_KEY not set — using stub LLM. "
        "All queries will route to portfolio_health."
    )
    return FakeLLMClient(
        default=ClassificationResult(
            intent="health_check",
            agent="portfolio_health",
            confidence=0.3,
            reasoning="stub LLM (no API key)",
        )
    )


def build_pipeline(
    *,
    settings: Settings | None = None,
    market_data: MarketData | None = None,
    llm: LLMClient | None = None,
) -> Pipeline:
    """Wire up the production pipeline. Each component is overridable so
    tests can inject fakes."""
    settings = settings or get_settings()
    fixtures_dir = Path(__file__).resolve().parents[2] / "fixtures"

    safety_guard = SafetyGuard()
    market_data = market_data or YFinanceMarketData()
    user_store = UserStore(fixtures_dir / "users")
    session_store = SessionStore(max_turns=max(2, settings.history_turns * 2))

    taxonomy = load_taxonomy(fixtures_dir / "test_queries" / "intent_classification.json")
    llm = llm or _build_llm_client(settings)
    classifier = IntentClassifier(
        llm=llm,
        agents=taxonomy,
        timeout_s=settings.classifier_timeout_s,
        history_turns=settings.history_turns,
    )
    registry = build_default_registry(taxonomy=taxonomy, market_data=market_data)

    return Pipeline(
        safety_guard=safety_guard,
        classifier=classifier,
        registry=registry,
        session_store=session_store,
        user_store=user_store,
        request_timeout_s=settings.request_timeout_s,
    )


def create_app(*, pipeline: Pipeline | None = None) -> FastAPI:
    """Build a FastAPI app. Tests pass a custom pipeline; production calls
    with no args."""
    app = FastAPI(
        title="Valura AI",
        description="Safety + classifier + portfolio-health agent + SSE.",
        version="0.1.0",
    )
    app.state.pipeline = pipeline or build_pipeline()
    app.include_router(v1_router)
    return app


# Module-level app instance for `uvicorn valura_ai.main:app`.
app = create_app()
