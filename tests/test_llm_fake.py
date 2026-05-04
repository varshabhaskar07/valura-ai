"""FakeLLMClient behavior tests.

These tests run entirely offline — no network, no OPENAI_API_KEY (the
top-level conftest deletes it). They verify the fake supports every
shape the classifier and agents will need: rule-based dispatch, schema
parsing, deterministic ordering, error simulation, call recording.
"""

from __future__ import annotations

import re

import pytest
from pydantic import BaseModel

from valura_ai.llm.client import (
    ChatMessage,
    LLMError,
    LLMParseError,
    LLMTimeoutError,
)
from valura_ai.llm.fakes import FakeLLMClient


class _Echo(BaseModel):
    text: str
    score: float = 0.0


class _Other(BaseModel):
    other: int


def _msg(text: str) -> list[ChatMessage]:
    return [ChatMessage(role="user", content=text)]


# ---------------------------------------------------------------------------
# Default response
# ---------------------------------------------------------------------------

async def test_default_response_returned_when_no_rules() -> None:
    fake = FakeLLMClient(default=_Echo(text="default"))
    out = await fake.structured(messages=_msg("anything"), schema=_Echo)
    assert out.text == "default"
    assert fake.call_count == 1


async def test_no_match_no_default_raises() -> None:
    fake = FakeLLMClient()
    with pytest.raises(LLMError, match="no rule matched"):
        await fake.structured(messages=_msg("hi"), schema=_Echo)


# ---------------------------------------------------------------------------
# Rule matching
# ---------------------------------------------------------------------------

async def test_substring_rule_matches_case_insensitively() -> None:
    fake = FakeLLMClient(default=_Echo(text="default"))
    fake.add_rule("microsoft", _Echo(text="msft"))
    out = await fake.structured(messages=_msg("tell me about MICROSOFT"), schema=_Echo)
    assert out.text == "msft"


async def test_regex_rule_supported() -> None:
    fake = FakeLLMClient(default=_Echo(text="default"))
    fake.add_rule(re.compile(r"\bAAPL\b", re.IGNORECASE), _Echo(text="apple"))
    out = await fake.structured(messages=_msg("what about AAPL today?"), schema=_Echo)
    assert out.text == "apple"


async def test_callable_rule_supported() -> None:
    fake = FakeLLMClient(default=_Echo(text="default"))
    fake.add_rule(lambda s: len(s.split()) > 5, _Echo(text="long"))
    out_long = await fake.structured(
        messages=_msg("this is a longer query for sure"), schema=_Echo
    )
    out_short = await fake.structured(messages=_msg("brief"), schema=_Echo)
    assert out_long.text == "long"
    assert out_short.text == "default"


async def test_first_matching_rule_wins() -> None:
    fake = FakeLLMClient(default=_Echo(text="default"))
    fake.add_rule("apple", _Echo(text="first"))
    fake.add_rule("apple", _Echo(text="second"))
    out = await fake.structured(messages=_msg("apple pie"), schema=_Echo)
    assert out.text == "first"


async def test_match_targets_latest_user_turn() -> None:
    """The fake should match against the LAST user turn, not the system prompt
    or earlier history. This mirrors how the real classifier prompts work."""
    fake = FakeLLMClient(default=_Echo(text="default"))
    fake.add_rule("microsoft", _Echo(text="msft"))
    out = await fake.structured(
        messages=[
            ChatMessage(role="system", content="You are a classifier"),
            ChatMessage(role="user", content="microsoft"),  # earlier history
            ChatMessage(role="assistant", content="MSFT is..."),
            ChatMessage(role="user", content="and apple"),  # current turn
        ],
        schema=_Echo,
    )
    # 'microsoft' rule should NOT fire — current turn is "and apple".
    assert out.text == "default"


# ---------------------------------------------------------------------------
# Error simulation
# ---------------------------------------------------------------------------

async def test_exception_response_is_raised() -> None:
    fake = FakeLLMClient()
    fake.add_rule("crash", LLMError("boom"))
    with pytest.raises(LLMError, match="boom"):
        await fake.structured(messages=_msg("crash now"), schema=_Echo)


async def test_timeout_can_be_simulated() -> None:
    fake = FakeLLMClient()
    fake.add_rule("slow", LLMTimeoutError("simulated 4s timeout"))
    with pytest.raises(LLMTimeoutError):
        await fake.structured(messages=_msg("slow path"), schema=_Echo)


async def test_default_can_be_an_exception() -> None:
    fake = FakeLLMClient(default=LLMError("api unavailable"))
    with pytest.raises(LLMError, match="api unavailable"):
        await fake.structured(messages=_msg("anything"), schema=_Echo)


async def test_factory_response_invoked_with_schema() -> None:
    """A factory lets a single rule satisfy multiple schemas, and gives tests
    a way to construct schema-dependent responses inline."""
    fake = FakeLLMClient()
    fake.add_rule("dynamic", lambda schema: schema(text=f"built-for-{schema.__name__}"))
    out = await fake.structured(messages=_msg("dynamic hi"), schema=_Echo)
    assert out.text == "built-for-_Echo"


# ---------------------------------------------------------------------------
# Schema enforcement
# ---------------------------------------------------------------------------

async def test_schema_mismatch_raises_parse_error() -> None:
    fake = FakeLLMClient()
    fake.add_rule("mismatch", _Other(other=1))  # wrong schema
    with pytest.raises(LLMParseError, match="not the requested schema"):
        await fake.structured(messages=_msg("mismatch please"), schema=_Echo)


async def test_factory_returning_wrong_schema_raises_parse_error() -> None:
    fake = FakeLLMClient()
    fake.add_rule("bad-factory", lambda _schema: _Other(other=1))
    with pytest.raises(LLMParseError, match="factory returned"):
        await fake.structured(messages=_msg("bad-factory"), schema=_Echo)


# ---------------------------------------------------------------------------
# Recording / introspection
# ---------------------------------------------------------------------------

async def test_calls_are_recorded_in_order() -> None:
    fake = FakeLLMClient(default=_Echo(text="ok"))
    await fake.structured(messages=_msg("first"), schema=_Echo)
    await fake.structured(messages=_msg("second"), schema=_Echo, timeout_s=2.5)
    await fake.structured(
        messages=_msg("third"), schema=_Echo, model="gpt-4.1"
    )
    assert fake.call_count == 3
    assert [c.user_text for c in fake.calls] == ["first", "second", "third"]
    assert fake.calls[1].timeout_s == 2.5
    assert fake.calls[2].model == "gpt-4.1"


async def test_reset_clears_calls_only() -> None:
    fake = FakeLLMClient(default=_Echo(text="ok"))
    fake.add_rule("hello", _Echo(text="hi"))
    await fake.structured(messages=_msg("hello"), schema=_Echo)
    assert fake.call_count == 1
    fake.reset()
    assert fake.call_count == 0
    # Rules survive reset:
    out = await fake.structured(messages=_msg("hello again"), schema=_Echo)
    assert out.text == "hi"


# ---------------------------------------------------------------------------
# Determinism: same input ⇒ same output
# ---------------------------------------------------------------------------

async def test_repeated_calls_are_deterministic() -> None:
    fake = FakeLLMClient()
    fake.add_rule("ping", _Echo(text="pong"))
    outs = [
        (await fake.structured(messages=_msg("ping"), schema=_Echo)).text
        for _ in range(20)
    ]
    assert outs == ["pong"] * 20
