"""OpenAILLMClient error-mapping tests.

We don't make real OpenAI calls — the conftest deletes OPENAI_API_KEY
to prove that. Instead, we mock the underlying SDK's
``client.beta.chat.completions.parse`` coroutine and verify that:

  - successful responses are returned typed
  - timeouts are re-raised as LLMTimeoutError
  - OpenAI exceptions are re-raised as LLMError
  - model refusals are re-raised as LLMError
  - parsed=None is re-raised as LLMParseError
  - per-call ``model`` and ``timeout_s`` overrides are propagated
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from valura_ai.llm.client import (
    ChatMessage,
    LLMError,
    LLMParseError,
    LLMTimeoutError,
    OpenAILLMClient,
)


class _Schema(BaseModel):
    text: str


def _build_response(*, parsed: object, refusal: str | None = None) -> SimpleNamespace:
    """Synthesise the shape the OpenAI SDK returns from .beta.chat.completions.parse()."""
    message = SimpleNamespace(parsed=parsed, refusal=refusal)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> OpenAILLMClient:
    # Constructing requires an api_key. Use a dummy — we mock the SDK before
    # any call. The conftest auto-deletes OPENAI_API_KEY from env so this
    # value cannot leak into a real call even if a test forgot to mock.
    c = OpenAILLMClient(api_key="sk-test-fake", model="gpt-4o-mini", default_timeout_s=2.0)
    # Replace the AsyncOpenAI instance with a mock that has the right shape.
    mock_sdk = MagicMock()
    mock_sdk.beta.chat.completions.parse = AsyncMock()
    c._client = mock_sdk
    return c


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

def test_constructor_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError, match="api_key"):
        OpenAILLMClient(api_key="", model="gpt-4o-mini")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_returns_parsed_pydantic_instance(client: OpenAILLMClient) -> None:
    expected = _Schema(text="hello")
    client._client.beta.chat.completions.parse.return_value = _build_response(parsed=expected)

    result = await client.structured(
        messages=[ChatMessage(role="user", content="hi")],
        schema=_Schema,
    )
    assert isinstance(result, _Schema)
    assert result.text == "hello"
    # Verify the SDK was called with our messages and schema.
    call_kwargs = client._client.beta.chat.completions.parse.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["response_format"] is _Schema
    assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]


async def test_per_call_model_override_is_propagated(client: OpenAILLMClient) -> None:
    client._client.beta.chat.completions.parse.return_value = _build_response(
        parsed=_Schema(text="x")
    )
    await client.structured(
        messages=[ChatMessage(role="user", content="hi")],
        schema=_Schema,
        model="gpt-4.1",
    )
    assert client._client.beta.chat.completions.parse.call_args.kwargs["model"] == "gpt-4.1"


# ---------------------------------------------------------------------------
# Timeout mapping
# ---------------------------------------------------------------------------

async def test_timeout_is_remapped_to_llm_timeout_error(client: OpenAILLMClient) -> None:
    async def slow(*_a, **_kw):
        await asyncio.sleep(0.5)
        return _build_response(parsed=_Schema(text="never"))

    client._client.beta.chat.completions.parse.side_effect = slow
    with pytest.raises(LLMTimeoutError, match="exceeded"):
        await client.structured(
            messages=[ChatMessage(role="user", content="hi")],
            schema=_Schema,
            timeout_s=0.05,
        )


async def test_default_timeout_used_when_call_omits_one(client: OpenAILLMClient) -> None:
    async def slow(*_a, **_kw):
        await asyncio.sleep(5.0)
        return _build_response(parsed=_Schema(text="never"))

    client._client.beta.chat.completions.parse.side_effect = slow
    # client default_timeout_s is 2.0, override that to a tiny value
    client._default_timeout_s = 0.05
    with pytest.raises(LLMTimeoutError):
        await client.structured(
            messages=[ChatMessage(role="user", content="hi")],
            schema=_Schema,
        )


# ---------------------------------------------------------------------------
# OpenAI errors
# ---------------------------------------------------------------------------

async def test_openai_exceptions_are_remapped(client: OpenAILLMClient) -> None:
    from openai import OpenAIError

    client._client.beta.chat.completions.parse.side_effect = OpenAIError("rate limit")
    with pytest.raises(LLMError, match="OpenAI API error"):
        await client.structured(
            messages=[ChatMessage(role="user", content="hi")],
            schema=_Schema,
        )


async def test_model_refusal_raises_llm_error(client: OpenAILLMClient) -> None:
    client._client.beta.chat.completions.parse.return_value = _build_response(
        parsed=None, refusal="I can't help with that."
    )
    with pytest.raises(LLMError, match="refused"):
        await client.structured(
            messages=[ChatMessage(role="user", content="hi")],
            schema=_Schema,
        )


async def test_none_parsed_raises_parse_error(client: OpenAILLMClient) -> None:
    client._client.beta.chat.completions.parse.return_value = _build_response(parsed=None)
    with pytest.raises(LLMParseError, match="no parsed content"):
        await client.structured(
            messages=[ChatMessage(role="user", content="hi")],
            schema=_Schema,
        )
