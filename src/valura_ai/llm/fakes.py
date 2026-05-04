"""FakeLLMClient — programmable, deterministic test double.

Implements the same ``LLMClient`` interface as the real client. Tests
build one of these, register rules mapping query patterns to responses
(or exceptions), and inject it wherever ``LLMClient`` is needed.

Deterministic by design: same fake + same input ⇒ same output, every
run. No timing, no network, no randomness. Tests are reproducible
without ``OPENAI_API_KEY`` and without flakiness.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Union

from pydantic import BaseModel

from .client import (
    ChatMessage,
    LLMClient,
    LLMError,
    LLMParseError,
    T,
)

# A matcher targets the text of the LATEST user turn in the prompt.
# Three forms supported:
#   - str       : case-insensitive substring match
#   - Pattern   : regex search
#   - Callable  : free-form predicate over the user content
Matcher = Union[str, re.Pattern[str], Callable[[str], bool]]

# A response is one of:
#   - BaseModel           : returned as-is (schema-checked at call time)
#   - Exception           : raised
#   - Callable[schema → BaseModel] : factory invoked with the requested schema
Response = Union[BaseModel, Exception, Callable[[type[BaseModel]], BaseModel]]


@dataclass(frozen=True)
class RecordedCall:
    """Snapshot of one structured() invocation. Useful for assertions."""
    messages: tuple[ChatMessage, ...]
    schema: type[BaseModel]
    model: str | None
    timeout_s: float | None

    @property
    def user_text(self) -> str:
        for m in reversed(self.messages):
            if m.role == "user":
                return m.content
        return ""


class FakeLLMClient(LLMClient):
    """Programmable LLMClient for tests.

    Example:
        from valura_ai.llm.client import LLMTimeoutError
        fake = FakeLLMClient(default=ClassificationResult(agent="support", ...))
        fake.add_rule("microsoft", ClassificationResult(agent="market_research", ...))
        fake.add_rule(re.compile(r"\\bAAPL\\b", re.I), apple_response)
        fake.add_rule("force_timeout", LLMTimeoutError("simulated"))

        result = await fake.structured(messages=[...], schema=ClassificationResult)
        assert fake.call_count == 1
    """

    __slots__ = ("_rules", "_default", "calls")

    def __init__(self, *, default: Response | None = None) -> None:
        self._rules: list[tuple[Matcher, Response]] = []
        self._default: Response | None = default
        self.calls: list[RecordedCall] = []

    # ----- configuration ---------------------------------------------------

    def add_rule(self, matcher: Matcher, response: Response) -> "FakeLLMClient":
        """Register a (matcher, response) pair. Earlier rules take priority."""
        self._rules.append((matcher, response))
        return self

    def set_default(self, response: Response) -> "FakeLLMClient":
        """Set the response used when no rule matches."""
        self._default = response
        return self

    # ----- introspection ---------------------------------------------------

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def reset(self) -> None:
        """Clear recorded calls (rules + default kept)."""
        self.calls.clear()

    # ----- LLMClient -------------------------------------------------------

    async def structured(
        self,
        *,
        messages: Sequence[ChatMessage],
        schema: type[T],
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> T:
        record = RecordedCall(
            messages=tuple(messages),
            schema=schema,
            model=model,
            timeout_s=timeout_s,
        )
        self.calls.append(record)

        for matcher, response in self._rules:
            if _match(matcher, record.user_text):
                return await _resolve(response, schema)

        if self._default is None:
            raise LLMError(
                f"FakeLLMClient: no rule matched user message "
                f"{record.user_text[:80]!r} and no default configured"
            )
        return await _resolve(self._default, schema)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _match(matcher: Matcher, text: str) -> bool:
    if isinstance(matcher, str):
        return matcher.lower() in text.lower()
    if isinstance(matcher, re.Pattern):
        return bool(matcher.search(text))
    return bool(matcher(text))


async def _resolve(response: Response, schema: type[BaseModel]) -> Any:
    if isinstance(response, Exception):
        raise response
    if isinstance(response, BaseModel):
        if not isinstance(response, schema):
            raise LLMParseError(
                f"FakeLLMClient: response of type {type(response).__name__} "
                f"is not the requested schema {schema.__name__}"
            )
        return response
    if callable(response):
        produced = response(schema)
        # Allow async factories — useful for tests that need to delay the
        # response (e.g. exercising the pipeline's request timeout).
        if asyncio.iscoroutine(produced):
            produced = await produced
        if not isinstance(produced, schema):
            raise LLMParseError(
                f"FakeLLMClient: factory returned {type(produced).__name__}, "
                f"expected {schema.__name__}"
            )
        return produced
    raise TypeError(
        f"FakeLLMClient: unsupported response type {type(response).__name__}; "
        "expected BaseModel, Exception, or Callable[schema → BaseModel]"
    )
