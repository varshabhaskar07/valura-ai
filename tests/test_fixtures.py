"""Schema validation for fixtures.

These tests don't evaluate classifier or guard quality — they only
guarantee that the fixture files are well-formed, balanced, and
internally consistent. If they pass, the downstream evaluation tests
in test_safety_guard.py and test_classifier.py have a sound input.

Failure of any of these tests usually means:
  - someone introduced a typo in a JSON file
  - the agent taxonomy was changed without updating the fixtures
  - someone added a query referencing an unknown agent or category
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

EXPECTED_USERS = {
    "user_001_aggressive_trader",
    "user_002_concentrated_nvda",
    "user_003_dividend_retiree",
    "user_004_empty",
    "user_005_global_multi_currency",
}

VALID_RISK_PROFILES = {"conservative", "moderate", "balanced", "aggressive"}
VALID_KYC_STATUS = {"verified", "pending", "rejected"}


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def test_all_five_users_present(fixtures_dir: Path) -> None:
    found = {p.stem for p in (fixtures_dir / "users").glob("*.json")}
    assert found == EXPECTED_USERS, f"missing or extra users: {found ^ EXPECTED_USERS}"


@pytest.mark.parametrize("user_id", sorted(EXPECTED_USERS))
def test_user_profile_well_formed(fixtures_dir: Path, user_id: str) -> None:
    data = json.loads((fixtures_dir / "users" / f"{user_id}.json").read_text())

    assert data["user_id"] == user_id
    assert data["kyc"]["status"] in VALID_KYC_STATUS
    assert data["risk_profile"] in VALID_RISK_PROFILES
    assert isinstance(data["base_currency"], str) and len(data["base_currency"]) == 3

    portfolio = data["portfolio"]
    assert isinstance(portfolio["positions"], list)
    assert isinstance(portfolio["cash"], dict)

    # Every position must have the full set of fields with correct types.
    for pos in portfolio["positions"]:
        assert isinstance(pos["ticker"], str) and pos["ticker"]
        assert isinstance(pos["exchange"], str) and pos["exchange"]
        assert isinstance(pos["quantity"], (int, float)) and pos["quantity"] > 0
        assert isinstance(pos["cost_basis_per_share"], (int, float))
        assert pos["cost_basis_per_share"] > 0
        assert isinstance(pos["currency"], str) and len(pos["currency"]) == 3


def test_user_004_is_empty_portfolio(fixtures_dir: Path) -> None:
    data = json.loads((fixtures_dir / "users" / "user_004_empty.json").read_text())
    assert data["portfolio"]["positions"] == []
    # But should still have cash so the BUILD-oriented response has something to suggest.
    assert sum(data["portfolio"]["cash"].values()) > 0


def test_user_005_has_multiple_currencies(fixtures_dir: Path) -> None:
    data = json.loads((fixtures_dir / "users" / "user_005_global_multi_currency.json").read_text())
    currencies = {p["currency"] for p in data["portfolio"]["positions"]}
    # Spec: at least 3 distinct currencies among positions.
    assert len(currencies) >= 3


def test_user_002_has_dominant_position(fixtures_dir: Path) -> None:
    """Concentration fixture only earns its name if one position actually dominates by cost basis."""
    data = json.loads((fixtures_dir / "users" / "user_002_concentrated_nvda.json").read_text())
    values = [p["quantity"] * p["cost_basis_per_share"] for p in data["portfolio"]["positions"]]
    top_share = max(values) / sum(values)
    assert top_share > 0.5, f"expected >50% concentration, got {top_share:.0%}"


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

def _conversation_files(fixtures_dir: Path) -> list[Path]:
    return sorted((fixtures_dir / "conversations").glob("*.json"))


def test_three_conversations_present(fixtures_dir: Path) -> None:
    assert len(_conversation_files(fixtures_dir)) == 3


def test_each_conversation_well_formed(fixtures_dir: Path) -> None:
    for path in _conversation_files(fixtures_dir):
        conv = json.loads(path.read_text())
        assert conv["conversation_id"] == path.stem
        assert conv["user_id"].startswith("user_")
        assert isinstance(conv["turns"], list) and len(conv["turns"]) >= 3
        assert all(t["role"] in {"user", "assistant"} for t in conv["turns"])
        assert isinstance(conv["assertions"], list) and conv["assertions"]
        for a in conv["assertions"]:
            assert 0 <= a["turn_index"] < len(conv["turns"])
            # Assertions must point at a USER turn (we route user input).
            assert conv["turns"][a["turn_index"]]["role"] == "user"


# ---------------------------------------------------------------------------
# Classification queries
# ---------------------------------------------------------------------------

def _load_classification(fixtures_dir: Path) -> dict:
    return json.loads((fixtures_dir / "test_queries" / "intent_classification.json").read_text())


def test_classification_taxonomy_present(fixtures_dir: Path) -> None:
    data = _load_classification(fixtures_dir)
    agents = data["agents"]
    assert "portfolio_health" in agents  # must include the implemented one
    assert len(agents) >= 6, "need at least 6 agents in taxonomy"
    assert len(agents) == len(set(agents)), "agent names must be unique"


def test_classification_query_count_meets_spec(fixtures_dir: Path) -> None:
    queries = _load_classification(fixtures_dir)["queries"]
    # Spec says ~60. We require comfortably more so per-agent buckets
    # are still meaningful after the hidden eval set is layered in.
    assert len(queries) >= 80


def test_classification_history_field_well_formed(fixtures_dir: Path) -> None:
    """If a query carries a `history` field, it must look like a real conversation prefix."""
    queries = _load_classification(fixtures_dir)["queries"]
    for q in queries:
        if "history" not in q:
            continue
        h = q["history"]
        assert isinstance(h, list) and h, q["id"]
        # Synthetic history must end with assistant — otherwise the
        # current user query is not "the next turn" the classifier sees.
        assert h[-1]["role"] == "assistant", q["id"]
        for turn in h:
            assert turn["role"] in {"user", "assistant"}, q["id"]
            assert isinstance(turn["text"], str) and turn["text"].strip(), q["id"]


def test_classification_has_history_dependent_queries(fixtures_dir: Path) -> None:
    """We need enough history-anchored queries to actually exercise session memory."""
    queries = _load_classification(fixtures_dir)["queries"]
    with_history = [q for q in queries if q.get("history")]
    assert len(with_history) >= 8, (
        f"only {len(with_history)} queries carry history; need >=8 to test "
        "follow-up resolution at scale"
    )


def test_classification_has_minimal_context_queries(fixtures_dir: Path) -> None:
    """At least 8 queries must be very short (<= 25 chars) to test minimal-context handling."""
    queries = _load_classification(fixtures_dir)["queries"]
    short = [q for q in queries if len(q["query"]) <= 25]
    assert len(short) >= 8, f"only {len(short)} short queries; need >=8"


def test_classification_queries_use_known_agents(fixtures_dir: Path) -> None:
    data = _load_classification(fixtures_dir)
    valid = set(data["agents"])
    bad = [q["id"] for q in data["queries"] if q["expected_agent"] not in valid]
    assert not bad, f"queries referencing unknown agents: {bad}"


def test_classification_distribution_is_balanced(fixtures_dir: Path) -> None:
    """Every agent in the taxonomy must have at least 5 queries.

    Anything less and a per-agent precision number isn't statistically
    meaningful on the public set.
    """
    data = _load_classification(fixtures_dir)
    counts = Counter(q["expected_agent"] for q in data["queries"])
    weak = {a: counts[a] for a in data["agents"] if counts[a] < 5}
    assert not weak, f"under-represented agents: {weak}"


def test_classification_ids_unique(fixtures_dir: Path) -> None:
    queries = _load_classification(fixtures_dir)["queries"]
    ids = [q["id"] for q in queries]
    assert len(ids) == len(set(ids))


def test_classification_queries_non_empty(fixtures_dir: Path) -> None:
    for q in _load_classification(fixtures_dir)["queries"]:
        assert isinstance(q["query"], str) and q["query"].strip()
        assert isinstance(q["expected_entities"], dict)


# ---------------------------------------------------------------------------
# Safety pairs
# ---------------------------------------------------------------------------

def _load_safety(fixtures_dir: Path) -> dict:
    return json.loads((fixtures_dir / "test_queries" / "safety_pairs.json").read_text())


def test_safety_pair_count_meets_spec(fixtures_dir: Path) -> None:
    pairs = _load_safety(fixtures_dir)["pairs"]
    # Spec says ~45. Expanded set should comfortably exceed.
    assert len(pairs) >= 70


def test_safety_has_enough_hard_pairs(fixtures_dir: Path) -> None:
    """Hard-difficulty rows are where keyword guards die; need a meaningful sample."""
    pairs = _load_safety(fixtures_dir)["pairs"]
    hard = [p for p in pairs if p.get("difficulty") == "hard"]
    assert len(hard) >= 15, f"only {len(hard)} hard-difficulty pairs; need >=15"


def test_safety_hard_set_is_balanced_block_vs_pass(fixtures_dir: Path) -> None:
    """Among hard pairs we need both block and pass examples — otherwise we're
    only testing one side of the precision/recall tradeoff."""
    pairs = _load_safety(fixtures_dir)["pairs"]
    hard = [p for p in pairs if p.get("difficulty") == "hard"]
    hard_block = sum(1 for p in hard if p["expected_block"])
    hard_pass = sum(1 for p in hard if not p["expected_block"])
    assert hard_block >= 5, f"only {hard_block} hard blocks"
    assert hard_pass >= 5, f"only {hard_pass} hard passes"


def test_safety_categories_present(fixtures_dir: Path) -> None:
    data = _load_safety(fixtures_dir)
    cats = data["categories"]
    must_have = {
        "insider_trading",
        "market_manipulation",
        "money_laundering",
        "guaranteed_returns",
        "reckless_advice",
    }
    assert must_have.issubset(set(cats))


def test_safety_pairs_use_known_categories(fixtures_dir: Path) -> None:
    data = _load_safety(fixtures_dir)
    valid = set(data["categories"])
    for pair in data["pairs"]:
        if pair["expected_block"]:
            assert pair["expected_category"] in valid, pair["id"]
        else:
            # Pass-through entries must NOT pin a category.
            assert pair["expected_category"] is None, pair["id"]


def test_safety_has_both_block_and_pass(fixtures_dir: Path) -> None:
    """Without both, recall and precision can't both be measured."""
    pairs = _load_safety(fixtures_dir)["pairs"]
    blocks = sum(1 for p in pairs if p["expected_block"])
    passes = sum(1 for p in pairs if not p["expected_block"])
    assert blocks >= 25
    assert passes >= 15


def test_safety_each_blocking_category_has_at_least_three_examples(fixtures_dir: Path) -> None:
    pairs = _load_safety(fixtures_dir)["pairs"]
    by_cat = Counter(p["expected_category"] for p in pairs if p["expected_block"])
    # Every blocking category we list should have enough samples to test recall.
    weak = {c: n for c, n in by_cat.items() if n < 3}
    assert not weak, f"under-represented blocking categories: {weak}"


def test_safety_ids_unique(fixtures_dir: Path) -> None:
    pairs = _load_safety(fixtures_dir)["pairs"]
    ids = [p["id"] for p in pairs]
    assert len(ids) == len(set(ids))
