"""One OpenAI-compatible model surface for the complete support graph."""
import hmac
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import get_settings
from src.graph import graph
from src.observability import flush, logger, trace_config


@asynccontextmanager
async def lifespan(app):
    yield
    flush()


app = FastAPI(title="Tuwaiq Technical Support Agent", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str = Field(max_length=20000)


class ChatCompletionRequest(BaseModel):
    model: str = "tuwaiq-tech-support-agent"
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    temperature: float = Field(default=0.2, ge=0, le=2)
    stream: bool = False


def authenticate(authorization: str | None = Header(default=None)):
    key = get_settings().api_key
    if not key:
        raise HTTPException(503, "API_KEY is not configured")
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token.encode(), key.encode()):
        raise HTTPException(401, "Invalid API key", headers={"WWW-Authenticate": "Bearer"})


@app.get("/health")
def health():
    # Liveness only: weights load lazily and external services are optional.
    return {"status": "ok"}


@app.get("/v1/models", dependencies=[Depends(authenticate)])
def models():
    return {"object": "list", "data": [{"id": get_settings().public_model, "object": "model",
                                         "created": 0, "owned_by": "tuwaiq"}]}


@app.post("/v1/chat/completions", dependencies=[Depends(authenticate)])
def chat_completions(req: ChatCompletionRequest):
    if req.model != get_settings().public_model:
        raise HTTPException(404, "Unknown model")
    user_messages = [m.content for m in req.messages if m.role == "user" and m.content.strip()]
    if not user_messages:
        raise HTTPException(400, "No nonempty user message provided")
    if sum(len(m.content) for m in req.messages) > 50000:
        raise HTTPException(413, "Conversation exceeds size limit")
    started = time.perf_counter()
    trace_id = uuid.uuid4().hex
    request_id = "chatcmpl-" + trace_id
    try:
        result = graph.invoke({"user_message": user_messages[-1], "request_id": request_id,
                               "trace_id": trace_id}, config=trace_config(request_id, trace_id))
    except Exception as exc:
        logger.error("request_id=%s error=%s latency=%.4f", request_id, type(exc).__name__,
                     time.perf_counter() - started)
        raise HTTPException(503, {"message": "Support backend unavailable; check model artifacts and database configuration",
                                  "request_id": request_id}) from exc
    latency = round(time.perf_counter() - started, 4)
    logger.info("request_id=%s route=%s latency=%.4f", request_id, result.get("route"), latency)
    response = {"id": request_id, "object": "chat.completion", "created": int(time.time()),
                "model": get_settings().public_model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": result["answer"]},
                             "finish_reason": "stop"}],
                # No single tokenizer spans this multi-model graph. Zero means unmetered.
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "system_metadata": {"route": result.get("route"), "intent": result.get("intent"),
                    "source": result.get("source"), "trace_id": trace_id, "latency_seconds": latency,
                    "router_latency_seconds": result.get("router_latency_seconds"),
                    "usage_metered": False, "escalated": result.get("escalate", False)}}
    if not req.stream:
        return response

    def events():
        # Buffered SSE supports WebUI defaults without pretending token-by-token inference.
        base = {key: response[key] for key in ("id", "created", "model")}
        base["object"] = "chat.completion.chunk"
        for delta, finish in (({"role": "assistant", "content": result["answer"]}, None), ({}, "stop")):
            chunk = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            yield "data: " + json.dumps(chunk) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
