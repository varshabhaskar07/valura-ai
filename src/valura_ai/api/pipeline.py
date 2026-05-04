"""Pipeline — safety guard → classifier → agent, streamed.

Single async generator. Each emission is an ``AgentEvent`` (or a
classification metadata event we synthesise after the classifier call).
The route layer wraps these in SSE events.

Order of operations per request:

  1. SafetyGuard.check(message). If blocked, emit an error event and
     return. Classifier never runs — the guard is the only authority
     on blocks (per assignment §Safety Precedence).
  2. Pull session history for this user (last N turns).
  3. Resolve user context for the classifier prompt + agent run.
  4. Classify (LLM with timeout; heuristic fallback). Emit a meta
     event with the structured ClassificationResult so the client can
     render routing info before tokens flow.
  5. Look up the agent in the registry. Stream its events through.
  6. Append the user's message + the assistant's prose to the session
     store so the next turn has context.

The whole thing runs under a single ``asyncio.timeout``. Hitting it
emits an error event and ends the stream — the route never returns a
500 for a slow agent.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ..agents.base import AgentEvent
from ..agents.registry import AgentRegistry
from ..classifier.classifier import IntentClassifier
from ..classifier.prompt import UserContext
from ..llm.client import ChatMessage
from ..portfolio.models import User
from ..safety.guard import SafetyGuard
from ..session.store import SessionStore
from .user_store import UserStore

logger = logging.getLogger(__name__)


# A meta event carrying the classification — produced by the pipeline,
# not by an agent. Keeps the agent contract clean.
@dataclass(frozen=True)
class _MetaEvent:
    classification: Any  # ClassificationResult; loose typed to avoid import cycle


class Pipeline:
    def __init__(
        self,
        *,
        safety_guard: SafetyGuard,
        classifier: IntentClassifier,
        registry: AgentRegistry,
        session_store: SessionStore,
        user_store: UserStore,
        request_timeout_s: float = 12.0,
    ) -> None:
        self._safety = safety_guard
        self._classifier = classifier
        self._registry = registry
        self._session = session_store
        self._users = user_store
        self._timeout_s = request_timeout_s

    async def run(
        self, *, user_id: str, message: str
    ) -> AsyncIterator[AgentEvent | _MetaEvent]:
        """Yield events for a single chat turn. Never raises — failures are
        emitted as error events."""
        try:
            async with asyncio.timeout(self._timeout_s):
                async for ev in self._run_inner(user_id=user_id, message=message):
                    yield ev
        except (asyncio.TimeoutError, TimeoutError):
            yield AgentEvent(
                type="error",
                text=f"Request exceeded the {self._timeout_s}s timeout. "
                     "Please try again or simplify the question.",
            )
        except Exception as e:  # last-resort safety net
            logger.exception("pipeline crashed")
            yield AgentEvent(
                type="error",
                text=f"Internal error: {type(e).__name__}",
            )

    async def _run_inner(
        self, *, user_id: str, message: str
    ) -> AsyncIterator[AgentEvent | _MetaEvent]:
        # 1) Safety guard — synchronous, sub-millisecond.
        verdict = self._safety.check(message)
        if verdict.is_blocked:
            yield AgentEvent(
                type="error",
                text=verdict.user_message or "Request blocked.",
            )
            return

        # 2) Session history + 3) user context.
        history = self._session.history(user_id)
        user: User | None = self._users.get(user_id)
        user_context = self._user_context(user) if user else None

        # 4) Classify.
        classification = await self._classifier.classify(
            query=message,
            history=history,
            user_context=user_context,
        )
        yield _MetaEvent(classification=classification)

        # 5) Route + stream.
        agent = self._registry.get(classification.agent)
        prose_buffer: list[str] = []
        async for ev in agent.stream(
            classification=classification,
            user=user,
            history=history,
        ):
            if ev.type == "token" and ev.text:
                prose_buffer.append(ev.text)
            yield ev

        # 6) Append turn to session memory (user + a synthesised assistant
        # turn so the next classification sees real conversation context).
        self._session.append(user_id, "user", message)
        if prose_buffer:
            assistant_text = "".join(prose_buffer).strip()
            # Cap at ~500 chars to keep prompt size bounded.
            if len(assistant_text) > 500:
                assistant_text = assistant_text[:500].rsplit(" ", 1)[0] + "…"
            self._session.append(user_id, "assistant", assistant_text)

    @staticmethod
    def _user_context(user: User) -> UserContext:
        return UserContext(
            user_id=user.user_id,
            risk_profile=user.risk_profile,
            base_currency=user.base_currency,
            holdings_tickers=tuple(p.ticker for p in user.portfolio.positions),
            kyc_status=user.kyc.status,
        )
