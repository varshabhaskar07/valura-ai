"""HTTP routes.

Single endpoint: ``POST /v1/chat`` returning Server-Sent Events.

The route function is intentionally thin — it parses the request body,
opens an SSE stream, and translates each pipeline event into the wire
format. All orchestration lives in ``Pipeline``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .pipeline import Pipeline, _MetaEvent
from .sse import done_event, from_agent_event, meta_event

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Body for POST /v1/chat."""

    message: str = Field(..., min_length=1, max_length=4000)
    user_id: str = Field(..., min_length=1, max_length=128)


router = APIRouter(prefix="/v1")


@router.post("/chat")
async def chat(req: ChatRequest, request: Request) -> EventSourceResponse:
    """Run one chat turn. Streams events via SSE.

    Event ordering on the happy path:
      meta → token+ → structured → done

    On safety block, classifier crash, agent failure, or timeout:
      [maybe meta] → error → (stream ends; no done event)
    """
    pipeline: Pipeline = request.app.state.pipeline

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        ended_with_error = False
        async for ev in pipeline.run(user_id=req.user_id, message=req.message):
            if isinstance(ev, _MetaEvent):
                yield meta_event(ev.classification)
                continue
            wire = from_agent_event(ev)
            yield wire
            if wire["event"] == "error":
                ended_with_error = True
                # An error event terminates the stream — nothing further would
                # be meaningful. We exit the generator; sse-starlette closes
                # the connection cleanly.
                return
        if not ended_with_error:
            yield done_event()

    return EventSourceResponse(event_stream())


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — separate from the chat endpoint so a slow agent
    never affects orchestrator health checks."""
    return {"status": "ok"}
