"""IntentClassifier — LLM call with deterministic heuristic fallback.

Single entry point: ``classify(query, history?, user_context?) -> ClassificationResult``.

Behaviour:
  - Build the messages prefix: system prompt (taxonomy + rules), then up
    to ``history_turns`` of prior user/assistant turns, then the current
    user query.
  - Call the LLM with a per-call timeout (defaults to ``Settings.classifier_timeout_s``).
  - On ANY ``LLMError`` (timeout, parse, refusal, transport), fall back to
    the heuristic. The heuristic emits the same schema, so callers cannot
    tell which branch produced the result.
  - The agent name is enforced against the taxonomy. If the LLM returns
    an unknown agent, we treat that as a parse failure and fall back.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

from ..llm.client import ChatMessage, LLMClient, LLMError
from .heuristic import HeuristicClassifier
from .prompt import UserContext, build_system_prompt
from .schema import ClassificationResult

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Owns the routing call: one LLM round-trip per request, with fallback."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        agents: list[str],
        timeout_s: float = 4.0,
        history_turns: int = 3,
        heuristic: HeuristicClassifier | None = None,
    ) -> None:
        if not agents:
            raise ValueError("agents taxonomy must be non-empty")
        self._llm = llm
        self._agents = list(agents)
        self._agent_set = set(self._agents)
        self._timeout_s = timeout_s
        self._history_turns = max(0, history_turns)
        self._heuristic = heuristic or HeuristicClassifier(agents=tuple(self._agents))

    async def classify(
        self,
        *,
        query: str,
        history: Sequence[ChatMessage] | None = None,
        user_context: UserContext | None = None,
    ) -> ClassificationResult:
        try:
            result = await self._llm_classify(query, history, user_context)
        except LLMError as e:
            logger.info("classifier LLM failed (%s); falling back to heuristic", e)
            return self._heuristic.classify(query, history)

        # Validate agent against taxonomy. An out-of-vocab agent is a
        # silent routing failure if we trust it — fall back instead.
        if result.agent not in self._agent_set:
            logger.info(
                "classifier returned unknown agent %r; falling back to heuristic",
                result.agent,
            )
            return self._heuristic.classify(query, history)
        return result

    async def _llm_classify(
        self,
        query: str,
        history: Sequence[ChatMessage] | None,
        user_context: UserContext | None,
    ) -> ClassificationResult:
        system = build_system_prompt(self._agents, user_context=user_context)
        messages: list[ChatMessage] = [ChatMessage(role="system", content=system)]
        if history:
            tail = list(history)[-(self._history_turns * 2):]  # turns ≈ user+assistant pairs
            messages.extend(tail)
        messages.append(ChatMessage(role="user", content=query))

        return await self._llm.structured(
            messages=messages,
            schema=ClassificationResult,
            timeout_s=self._timeout_s,
        )


# ---------------------------------------------------------------------------
# Taxonomy loading
# ---------------------------------------------------------------------------

def load_taxonomy(path: Path | None = None) -> list[str]:
    """Load the agents taxonomy from the fixtures file.

    Production code calls this at startup; tests can pass a specific path
    or supply the agents list directly to ``IntentClassifier``.
    """
    if path is None:
        # Look up two parents from this file: src/valura_ai/classifier/ -> repo root
        path = (
            Path(__file__).resolve().parents[3]
            / "fixtures"
            / "test_queries"
            / "intent_classification.json"
        )
    data = json.loads(path.read_text())
    agents = data["agents"]
    if not isinstance(agents, list) or not all(isinstance(a, str) for a in agents):
        raise ValueError(f"taxonomy at {path} is malformed")
    return agents
