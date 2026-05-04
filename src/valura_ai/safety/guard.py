"""SafetyGuard — pure-Python pre-LLM filter.

The guard scores each query against the category patterns in
``patterns.py`` and combines the result with cross-category signals
(methodology, evasion, educational framing). No LLM, no I/O. Designed
to complete well under 10 ms for any realistic query length.

Decision precedence (first match wins):

  R1) phrase ≥1 in best category AND methodology marker present  → BLOCK
  R2) phrase ≥1 in best category AND evasion marker present      → BLOCK
  R3) score ≥ 2.0 in best category AND no educational marker     → BLOCK
  R4) otherwise                                                   → PASS

Score is computed per category as ``phrase_hits * 1 + intent_hits * 2``
(intent matches both substring and regex patterns). The "best category"
is whichever has the highest score; ties broken by the order in
``ALL_CATEGORIES``.

Why educational framing only excuses pure phrase matches:
  A user asking "what is insider trading" cites the topic but isn't
  asking how to do it. Conversely, "explain insider trading and how
  it's done" cites the topic AND requests methodology — the methodology
  half is the harmful part, so educational framing must not override
  it. R1 enforces this asymmetry directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .patterns import (
    ALL_CATEGORIES,
    EDUCATIONAL_MARKERS,
    EVASION_MARKERS,
    METHODOLOGY_MARKERS,
    CategoryPatterns,
)


@dataclass(frozen=True)
class SafetyResult:
    """Outcome of a single guard evaluation."""

    is_blocked: bool
    category: str | None
    reason: str
    user_message: str | None  # refusal text when blocked; None when passing


# Sentinel for the pass-through case (kept out of the blocked path so the
# caller can identify "no signal at all" by `category is None`).
_PASS = SafetyResult(
    is_blocked=False,
    category=None,
    reason="no harmful-intent signals detected",
    user_message=None,
)


class SafetyGuard:
    """Stateless safety evaluator. Constructable per-process; calls are
    independent and thread-safe."""

    __slots__ = ("_categories",)

    def __init__(self, categories: tuple[CategoryPatterns, ...] = ALL_CATEGORIES) -> None:
        self._categories = categories

    def check(self, query: str) -> SafetyResult:
        if not query or not query.strip():
            return _PASS

        # Single normalisation pass: lowercase + collapse non-ascii dashes.
        # Lowercasing is enough for substring matching; we deliberately do
        # not strip punctuation because the patterns include hyphens and
        # apostrophes ("c-suite", "won't").
        q = query.lower().replace("—", "-").replace("–", "-")

        has_methodology = _any_in(METHODOLOGY_MARKERS, q)
        has_evasion = _any_in(EVASION_MARKERS, q)
        has_educational = _any_in(EDUCATIONAL_MARKERS, q)

        best: CategoryPatterns | None = None
        best_score = 0.0
        best_phrase_hits = 0
        best_intent_hits = 0

        for cat in self._categories:
            phrase_hits = sum(1 for p in cat.phrases if p in q)
            intent_hits = sum(1 for p in cat.intent_phrases if p in q)
            intent_hits += sum(1 for r in cat.intent_regex if r.search(q))
            if phrase_hits == 0 and intent_hits == 0:
                continue
            score = phrase_hits * 1.0 + intent_hits * 2.0
            if score > best_score:
                best, best_score = cat, score
                best_phrase_hits, best_intent_hits = phrase_hits, intent_hits

        if best is None:
            return _PASS

        has_category_signal = best_phrase_hits > 0 or best_intent_hits > 0

        # R1) phrase + methodology in same category context → block.
        if has_category_signal and has_methodology:
            return SafetyResult(
                is_blocked=True,
                category=best.name,
                reason=f"{best.name}: category phrase + methodology marker",
                user_message=best.refusal,
            )

        # R2) phrase + evasion → block.
        if has_category_signal and has_evasion:
            return SafetyResult(
                is_blocked=True,
                category=best.name,
                reason=f"{best.name}: category phrase + evasion marker",
                user_message=best.refusal,
            )

        # R3) score ≥ 2 and no educational framing → block.
        if best_score >= 2.0 and not has_educational:
            return SafetyResult(
                is_blocked=True,
                category=best.name,
                reason=(
                    f"{best.name}: cumulative signals "
                    f"(phrases={best_phrase_hits}, intent={best_intent_hits}, score={best_score:.1f})"
                ),
                user_message=best.refusal,
            )

        # R4) pass — defensive question or weak signal.
        return SafetyResult(
            is_blocked=False,
            category=None,
            reason=(
                f"weak signal in {best.name} "
                f"(score={best_score:.1f}, educational={has_educational})"
            ),
            user_message=None,
        )


def _any_in(needles: tuple[str, ...], haystack: str) -> bool:
    """Faster than ``any(n in s for n in t)`` only marginally, but here for
    clarity — short-circuits and avoids the generator overhead."""
    for n in needles:
        if n in haystack:
            return True
    return False
