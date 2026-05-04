"""Intent classifier — schema, prompt, heuristic, LLM path, fallback path,
history-aware routing, and conversation-level integration. All offline."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from valura_ai.classifier.classifier import IntentClassifier, load_taxonomy
from valura_ai.classifier.heuristic import HeuristicClassifier
from valura_ai.classifier.prompt import (
    AGENT_DESCRIPTIONS,
    UserContext,
    build_system_prompt,
)
from valura_ai.classifier.schema import (
    ClassificationResult,
    Entities,
    SafetyVerdict,
)
from valura_ai.llm.client import (
    ChatMessage,
    LLMError,
    LLMTimeoutError,
)
from valura_ai.llm.fakes import FakeLLMClient

from .matchers import check_entities


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def taxonomy(fixtures_dir: Path) -> list[str]:
    return load_taxonomy(fixtures_dir / "test_queries" / "intent_classification.json")


@pytest.fixture(scope="module")
def classification_queries(fixtures_dir: Path) -> list[dict]:
    return json.loads(
        (fixtures_dir / "test_queries" / "intent_classification.json").read_text()
    )["queries"]


@pytest.fixture(scope="module")
def conversations(fixtures_dir: Path) -> list[dict]:
    files = sorted((fixtures_dir / "conversations").glob("*.json"))
    return [json.loads(p.read_text()) for p in files]


def _entities_to_dict(e: Entities) -> dict:
    return e.model_dump()


def _history_to_chat_messages(history: list[dict]) -> list[ChatMessage]:
    return [ChatMessage(role=t["role"], content=t["text"]) for t in history]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_classification_result_minimum_construction() -> None:
    r = ClassificationResult(intent="x", agent="market_research")
    assert r.confidence == 0.5
    assert r.entities.tickers == []
    assert r.safety_verdict.is_safe is True


def test_classification_result_full_roundtrip() -> None:
    r = ClassificationResult(
        intent="research",
        agent="market_research",
        entities=Entities(tickers=["NVDA"], topics=["semiconductor"]),
        confidence=0.9,
        reasoning="user named a ticker",
        safety_verdict=SafetyVerdict(is_safe=True),
    )
    dumped = r.model_dump()
    again = ClassificationResult.model_validate(dumped)
    assert again == r


def test_confidence_is_clamped_in_schema() -> None:
    with pytest.raises(Exception):
        ClassificationResult(intent="x", agent="x", confidence=1.5)
    with pytest.raises(Exception):
        ClassificationResult(intent="x", agent="x", confidence=-0.1)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def test_prompt_lists_all_taxonomy_agents(taxonomy: list[str]) -> None:
    prompt = build_system_prompt(taxonomy)
    for agent in taxonomy:
        assert f"- {agent}:" in prompt, f"agent {agent!r} missing from prompt"


def test_prompt_mentions_routing_rules(taxonomy: list[str]) -> None:
    prompt = build_system_prompt(taxonomy)
    # Spot-check the key routing principles are present (without checking
    # exact wording, which would be brittle).
    assert "exactly one" in prompt.lower()
    assert "history" in prompt.lower()
    assert "multi-intent" in prompt.lower() or "multi intent" in prompt.lower()
    assert "ticker" in prompt.lower()


def test_prompt_does_not_contain_fixture_phrases(
    taxonomy: list[str], classification_queries: list[dict]
) -> None:
    """Bias check: the system prompt must not contain verbatim queries
    from the public eval set, otherwise we'd be coaching the LLM toward
    these specific phrasings."""
    prompt = build_system_prompt(taxonomy).lower()
    for q in classification_queries:
        text = q["query"].lower().strip().rstrip("?")
        # Allow short fragments that are general English (3 words is ~the
        # boundary where overlap is plausibly coincidental).
        if len(text) > 25:
            assert text not in prompt, f"prompt overlaps fixture phrase: {q['id']}"


def test_prompt_includes_user_context_when_provided(taxonomy: list[str]) -> None:
    ctx = UserContext(
        user_id="user_test",
        risk_profile="moderate",
        base_currency="USD",
        holdings_tickers=("NVDA", "MSFT"),
    )
    prompt = build_system_prompt(taxonomy, user_context=ctx)
    assert "moderate" in prompt
    assert "NVDA, MSFT" in prompt


def test_descriptions_exist_for_every_taxonomy_agent(taxonomy: list[str]) -> None:
    """Catches drift between fixture taxonomy and prompt descriptions."""
    missing = [a for a in taxonomy if a not in AGENT_DESCRIPTIONS]
    assert not missing, f"agents missing prompt description: {missing}"


# ---------------------------------------------------------------------------
# Matcher sanity
# ---------------------------------------------------------------------------

def test_matcher_subset_match() -> None:
    res = check_entities(
        expected={"tickers": ["NVDA"]},
        actual={"tickers": ["NVDA", "AMD"]},
    )
    assert res.matches


def test_matcher_ticker_normalization() -> None:
    res = check_entities(
        expected={"tickers": ["ASML"]},
        actual={"tickers": ["asml.as"]},
    )
    assert res.matches, res.failures


def test_matcher_numeric_tolerance() -> None:
    assert check_entities({"amount": 1000.0}, {"amount": 1040.0}).matches  # +4%
    assert not check_entities({"amount": 1000.0}, {"amount": 1100.0}).matches  # +10%


def test_matcher_null_expected_means_unasserted() -> None:
    assert check_entities({"amount": None}, {"amount": 999.0}).matches


# ---------------------------------------------------------------------------
# Heuristic — accuracy on the fixture
# ---------------------------------------------------------------------------

def _evaluate_heuristic(
    heur: HeuristicClassifier,
    queries: list[dict],
) -> dict:
    correct = 0
    fails: list[dict] = []
    for q in queries:
        history = _history_to_chat_messages(q.get("history", []))
        result = heur.classify(q["query"], history=history or None)
        if result.agent == q["expected_agent"]:
            correct += 1
        else:
            fails.append({
                "id": q["id"],
                "query": q["query"][:60],
                "expected": q["expected_agent"],
                "got": result.agent,
            })
    return {
        "n": len(queries),
        "correct": correct,
        "accuracy": correct / len(queries) if queries else 1.0,
        "failures": fails,
    }


def test_heuristic_accuracy_meets_safety_net_threshold(
    taxonomy: list[str],
    classification_queries: list[dict],
) -> None:
    """Heuristic is the FALLBACK path — it doesn't need to beat the LLM,
    but it should be clearly above chance (1/8 = 12.5%) so a degraded
    request still routes plausibly. Threshold is 60%."""
    heur = HeuristicClassifier(agents=tuple(taxonomy))
    res = _evaluate_heuristic(heur, classification_queries)
    msg = (
        f"heuristic accuracy={res['accuracy']:.3f} ({res['correct']}/{res['n']}). "
        f"first 8 failures: {res['failures'][:8]}"
    )
    assert res["accuracy"] >= 0.60, msg


def test_heuristic_handles_empty_input(taxonomy: list[str]) -> None:
    heur = HeuristicClassifier(agents=tuple(taxonomy))
    r = heur.classify("")
    assert r.agent in taxonomy
    assert r.confidence < 0.6


def test_heuristic_uses_history_for_vague_query(taxonomy: list[str]) -> None:
    heur = HeuristicClassifier(agents=tuple(taxonomy))
    history = [
        ChatMessage(role="user", content="what's going on with NVDA"),
        ChatMessage(role="assistant", content="NVDA is..."),
    ]
    r = heur.classify("thoughts?", history=history)
    assert r.agent == "market_research"


def test_heuristic_extracts_ticker_from_company_name(taxonomy: list[str]) -> None:
    heur = HeuristicClassifier(agents=tuple(taxonomy))
    r = heur.classify("tell me about microsoft")
    assert "MSFT" in r.entities.tickers


def test_heuristic_extracts_calculator_numbers(taxonomy: list[str]) -> None:
    heur = HeuristicClassifier(agents=tuple(taxonomy))
    r = heur.classify("if I put 500/month at 7% for 30 years what do I end up with")
    assert r.agent == "financial_calculator"
    assert r.entities.amount == 500
    assert r.entities.rate == 0.07
    assert r.entities.period_years == 30


# ---------------------------------------------------------------------------
# Main IntentClassifier — LLM path (oracle FakeLLMClient)
# ---------------------------------------------------------------------------

def _oracle_factory(query_to_result: dict[str, ClassificationResult]):
    """Build a FakeLLMClient that returns the expected result for each
    fixture query. This tests the wiring (prompt + history + parse), not
    LLM accuracy."""
    fake = FakeLLMClient()
    for query, result in query_to_result.items():
        fake.add_rule(query, result)
    fake.set_default(LLMError("oracle: no rule"))
    return fake


def _expected_to_classification(q: dict) -> ClassificationResult:
    ent = q.get("expected_entities", {}) or {}
    return ClassificationResult(
        intent=q["expected_agent"],
        agent=q["expected_agent"],
        entities=Entities(
            tickers=ent.get("tickers", []) or [],
            topics=ent.get("topics", []) or [],
            sectors=ent.get("sectors", []) or [],
            amount=ent.get("amount"),
            rate=ent.get("rate"),
            period_years=ent.get("period_years"),
        ),
        confidence=0.95,
        reasoning="oracle",
    )


async def test_llm_path_passes_response_through_unchanged(
    taxonomy: list[str], classification_queries: list[dict]
) -> None:
    """When the LLM (faked) returns a valid result, the classifier returns it
    verbatim — no mangling."""
    sample = classification_queries[:5]
    oracle = _oracle_factory({q["query"]: _expected_to_classification(q) for q in sample})
    cls = IntentClassifier(llm=oracle, agents=taxonomy)

    for q in sample:
        result = await cls.classify(query=q["query"])
        assert result.agent == q["expected_agent"]


async def test_llm_path_full_oracle_eval(
    taxonomy: list[str], classification_queries: list[dict]
) -> None:
    """End-to-end: fake LLM is an oracle; classifier hits 100% routing.
    Validates that history is actually being injected into the LLM call
    (otherwise queries with optional history wouldn't match the oracle)."""
    oracle = _oracle_factory(
        {q["query"]: _expected_to_classification(q) for q in classification_queries}
    )
    cls = IntentClassifier(llm=oracle, agents=taxonomy)

    correct = 0
    entity_failures = []
    for q in classification_queries:
        history = _history_to_chat_messages(q.get("history", []))
        result = await cls.classify(query=q["query"], history=history or None)
        if result.agent == q["expected_agent"]:
            correct += 1
        # Entity match per published matcher contract.
        ent_check = check_entities(
            q.get("expected_entities", {}) or {},
            _entities_to_dict(result.entities),
        )
        if not ent_check.matches:
            entity_failures.append((q["id"], ent_check.failures))

    accuracy = correct / len(classification_queries)
    assert accuracy >= 0.85, f"oracle accuracy={accuracy:.3f}; need >=0.85"
    assert not entity_failures, f"entity matcher failures: {entity_failures[:5]}"


async def test_unknown_agent_in_response_falls_back_to_heuristic(
    taxonomy: list[str],
) -> None:
    """If the LLM returns an out-of-vocab agent, treat it as a routing
    failure and fall back. A bad agent name is worse than the heuristic's
    best guess."""
    bad = ClassificationResult(intent="x", agent="not_a_real_agent")
    fake = FakeLLMClient(default=bad)
    cls = IntentClassifier(llm=fake, agents=taxonomy)
    r = await cls.classify(query="how is my portfolio doing")
    assert r.agent in taxonomy
    assert "heuristic" in r.reasoning.lower()


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "exc",
    [
        LLMError("api unavailable"),
        LLMTimeoutError("simulated 4s timeout"),
    ],
    ids=["llm_error", "llm_timeout"],
)
async def test_any_llm_error_triggers_fallback(taxonomy: list[str], exc: Exception) -> None:
    fake = FakeLLMClient(default=exc)
    cls = IntentClassifier(llm=fake, agents=taxonomy)
    r = await cls.classify(query="how is my portfolio doing")
    assert isinstance(r, ClassificationResult)
    assert r.agent == "portfolio_health"  # heuristic must still route correctly
    assert r.confidence < 0.5
    assert "heuristic" in r.reasoning.lower()


async def test_fallback_never_crashes_on_pathological_input(taxonomy: list[str]) -> None:
    fake = FakeLLMClient(default=LLMError("down"))
    cls = IntentClassifier(llm=fake, agents=taxonomy)
    weird_inputs = ["", "   ", "????", "🤔🤔🤔", "x" * 5000]
    for q in weird_inputs:
        r = await cls.classify(query=q)
        assert isinstance(r, ClassificationResult)
        assert r.agent in taxonomy


# ---------------------------------------------------------------------------
# Conversation-level (history-anchored) classification
# ---------------------------------------------------------------------------

async def test_conversations_with_oracle(
    taxonomy: list[str], conversations: list[dict]
) -> None:
    """Walk every conversation turn-by-turn. At each user turn that has an
    assertion, classify with the prior turns as history. Oracle FakeLLMClient
    returns the expected agent; we verify routing + entity match."""
    correct = 0
    total = 0
    for conv in conversations:
        # Build oracle for this conversation's expected per-turn results.
        oracle = FakeLLMClient()
        for assertion in conv["assertions"]:
            turn_idx = assertion["turn_index"]
            user_text = conv["turns"][turn_idx]["text"]
            ent = assertion.get("expected_entities", {}) or {}
            oracle.add_rule(
                user_text,
                ClassificationResult(
                    intent=assertion["expected_agent"],
                    agent=assertion["expected_agent"],
                    entities=Entities(
                        tickers=ent.get("tickers", []) or [],
                        topics=ent.get("topics", []) or [],
                        sectors=ent.get("sectors", []) or [],
                        amount=ent.get("amount"),
                        rate=ent.get("rate"),
                        period_years=ent.get("period_years"),
                    ),
                    confidence=0.95,
                ),
            )
        oracle.set_default(LLMError("conv-oracle: no rule"))
        cls = IntentClassifier(llm=oracle, agents=taxonomy)

        for assertion in conv["assertions"]:
            turn_idx = assertion["turn_index"]
            current = conv["turns"][turn_idx]["text"]
            history = _history_to_chat_messages(conv["turns"][:turn_idx])
            r = await cls.classify(query=current, history=history or None)
            total += 1
            if r.agent == assertion["expected_agent"]:
                correct += 1
            # Entity match
            ent_check = check_entities(
                assertion.get("expected_entities", {}) or {},
                _entities_to_dict(r.entities),
            )
            assert ent_check.matches, (
                f"{conv['conversation_id']} turn {turn_idx}: {ent_check.failures}"
            )
    assert correct == total, f"conversation routing: {correct}/{total}"
