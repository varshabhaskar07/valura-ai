"""Base contract for streaming agents.

Every agent is an async generator yielding ``AgentEvent``s. The HTTP
layer in Step 9 maps these onto SSE events. Keeping the agent contract
event-typed (rather than a raw string stream) lets us emit:

  - prose tokens for the user
  - one ``structured`` payload for downstream consumers (logging, history,
    structured-output requirement from the assignment)
  - one-off ``info`` events (e.g. "looking at your portfolio…") to keep
    first-token latency under the 2s budget when the heavy compute step
    takes a beat
  - ``error`` events when the agent can't run (missing context,
    unimplemented stub) — the pipeline turns these into structured SSE
    error events instead of stack traces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import BaseModel

from ..classifier.schema import ClassificationResult
from ..llm.client import ChatMessage
from ..portfolio.models import User


@dataclass(frozen=True)
class AgentEvent:
    """One unit emitted by an agent during streaming."""

    type: Literal["info", "token", "structured", "error"]
    text: str | None = None
    data: BaseModel | None = None


class BaseAgent(ABC):
    """Streaming agent. Subclasses set ``name`` and implement ``stream``."""

    name: ClassVar[str]

    @abstractmethod
    def stream(
        self,
        *,
        classification: ClassificationResult,
        user: User | None = None,
        history: Sequence[ChatMessage] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Run the agent. Returns an async iterator of events."""
