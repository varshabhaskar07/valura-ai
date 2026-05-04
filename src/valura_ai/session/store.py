"""In-memory session memory: last N turns per user.

Defended choice (per the assignment, "Persistence is your choice ...
justify your pick in the README"): in-memory is correct for a 3-day
single-process demo. The interface — ``SessionStore`` — is the only
thing the pipeline depends on, so a Postgres or Redis-backed
implementation is a drop-in replacement when the system grows past one
process.

Concurrency: each user_id's deque is touched by at most one in-flight
request in this build (no parallelism per user). If that changes, wrap
the deque in a per-key lock — the interface doesn't have to.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Literal

from ..llm.client import ChatMessage


class SessionStore:
    """Per-user FIFO of ``ChatMessage``s, bounded length."""

    def __init__(self, *, max_turns: int = 10) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self._max_turns = max_turns
        self._turns: dict[str, deque[ChatMessage]] = {}

    def history(self, user_id: str) -> list[ChatMessage]:
        """Return prior turns for this user (newest last). Safe to call for
        unknown users — returns an empty list."""
        d = self._turns.get(user_id)
        return list(d) if d else []

    def append(
        self,
        user_id: str,
        role: Literal["user", "assistant"],
        content: str,
    ) -> None:
        """Record a turn. Overflow drops the oldest entry."""
        if not content:
            return
        d = self._turns.get(user_id)
        if d is None:
            d = deque(maxlen=self._max_turns)
            self._turns[user_id] = d
        d.append(ChatMessage(role=role, content=content))

    def append_many(self, user_id: str, turns: Sequence[ChatMessage]) -> None:
        for t in turns:
            self.append(user_id, t.role, t.content)  # type: ignore[arg-type]

    def reset(self, user_id: str | None = None) -> None:
        """Clear one user's history, or all when called with no argument."""
        if user_id is None:
            self._turns.clear()
        else:
            self._turns.pop(user_id, None)
