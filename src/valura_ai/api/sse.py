"""SSE event encoding.

We use sse-starlette for transport (handles keep-alives, disconnect
detection, content-type) but format the per-event payloads ourselves
so the event names + JSON shape are explicit and easy to test.

Event types emitted by the pipeline:

  meta       — one event after safety + classifier; carries the
               ClassificationResult JSON. Lets the client begin
               rendering routing info before the agent stream starts.
  token      — one or more events; each carries a chunk of prose text.
  structured — one event at the end of the stream; carries the agent's
               final structured payload (PortfolioHealthReport, StubResponse, ...).
  error      — one event when the request can't proceed (safety blocked,
               classifier hard-failed, agent raised, request timed out).
               After an error event the stream ends.
  done       — one terminal event with no data, sent on success.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from ..agents.base import AgentEvent
from ..classifier.schema import ClassificationResult


def event_dict(name: str, data: Any) -> dict[str, str]:
    """Format one SSE event as the dict shape ``EventSourceResponse`` accepts."""
    if isinstance(data, BaseModel):
        payload = data.model_dump_json()
    elif isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, separators=(",", ":"))
    return {"event": name, "data": payload}


def meta_event(classification: ClassificationResult) -> dict[str, str]:
    return event_dict("meta", classification)


def error_event(message: str, *, category: str | None = None) -> dict[str, str]:
    body: dict[str, str] = {"message": message}
    if category:
        body["category"] = category
    return event_dict("error", body)


def done_event() -> dict[str, str]:
    return {"event": "done", "data": ""}


def from_agent_event(ev: AgentEvent) -> dict[str, str]:
    """Translate an internal AgentEvent into an SSE event dict."""
    if ev.type == "token":
        return event_dict("token", ev.text or "")
    if ev.type == "structured":
        if ev.data is None:
            return event_dict("structured", {})
        return event_dict("structured", ev.data)
    if ev.type == "error":
        return error_event(ev.text or "agent error")
    # info — not currently used downstream of the agent, but mirror it
    # so a future agent that emits info events is forwarded correctly.
    if ev.data is not None:
        return event_dict("info", ev.data)
    return event_dict("info", ev.text or "")
