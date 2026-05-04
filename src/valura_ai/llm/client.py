"""LLM client abstraction.

A single async interface — ``LLMClient.structured(messages, schema, ...)``
— for getting Pydantic-validated structured outputs out of an LLM. Two
implementations live in this package:

  - ``OpenAILLMClient`` (this file)  → production
  - ``FakeLLMClient`` (``fakes.py``)  → tests / offline development

Tests must never hit the real client; ``conftest.py`` deletes
``OPENAI_API_KEY`` from the environment so accidental construction
of ``OpenAILLMClient`` fails loudly.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChatMessage:
    """A single turn in the prompt to the LLM."""
    role: Literal["system", "user", "assistant"]
    content: str


class LLMError(Exception):
    """Base error for any LLMClient failure."""


class LLMTimeoutError(LLMError):
    """The per-request timeout was exceeded."""


class LLMParseError(LLMError):
    """The model returned content that did not validate against the schema."""


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Async client returning structured (Pydantic) outputs.

    The abstraction is deliberately narrow: one method, structured outputs
    only. We do not expose chat completion or streaming through this
    interface — a streaming agent owns its own response shape and has a
    different contract. This client exists for the classifier and any other
    one-shot, one-schema call site.
    """

    @abstractmethod
    async def structured(
        self,
        *,
        messages: Sequence[ChatMessage],
        schema: type[T],
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> T:
        """Issue one structured-output completion call.

        Args:
            messages: Conversation prefix to send to the model.
            schema: Pydantic model class the response must conform to.
            model: Optional per-call model override.
            timeout_s: Optional per-call timeout override.

        Raises:
            LLMTimeoutError: per-request timeout was hit.
            LLMParseError:   response could not be coerced to ``schema``.
            LLMError:        any other client-side failure (auth, rate limit,
                             refusal, connectivity).
        """
        ...


# ---------------------------------------------------------------------------
# Production implementation
# ---------------------------------------------------------------------------

class OpenAILLMClient(LLMClient):
    """Thin wrapper around the OpenAI Python SDK's structured outputs API.

    All errors are mapped into the ``LLMError`` hierarchy so callers (the
    classifier, agents) never need to import openai exceptions directly.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        default_timeout_s: float = 6.0,
    ) -> None:
        if not api_key:
            raise ValueError(
                "OpenAILLMClient requires a non-empty api_key. "
                "Tests should use FakeLLMClient instead."
            )
        # Imported lazily so test runs that exclusively use the fake never
        # require the openai package to load successfully.
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._default_timeout_s = default_timeout_s

    async def structured(
        self,
        *,
        messages: Sequence[ChatMessage],
        schema: type[T],
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> T:
        from openai import OpenAIError

        timeout = timeout_s if timeout_s is not None else self._default_timeout_s
        chosen_model = model or self._model
        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        try:
            resp = await asyncio.wait_for(
                self._client.beta.chat.completions.parse(
                    model=chosen_model,
                    messages=api_messages,
                    response_format=schema,
                ),
                timeout=timeout,
            )
        except (asyncio.TimeoutError, TimeoutError) as e:
            raise LLMTimeoutError(
                f"LLM call exceeded {timeout}s (model={chosen_model})"
            ) from e
        except OpenAIError as e:
            raise LLMError(f"OpenAI API error: {e}") from e

        choice = resp.choices[0]
        if getattr(choice.message, "refusal", None):
            raise LLMError(f"model refused: {choice.message.refusal}")

        parsed = choice.message.parsed
        if parsed is None:
            raise LLMParseError("model returned no parsed content")
        if not isinstance(parsed, schema):
            # The SDK should validate against ``schema`` for us, but if the
            # response shape ever drifts (e.g. SDK upgrade) we'd rather
            # surface a typed parse error than a confusing TypeError later.
            try:
                parsed = schema.model_validate(parsed)
            except ValidationError as e:
                raise LLMParseError(
                    f"failed to validate against {schema.__name__}: {e}"
                ) from e
        return parsed
