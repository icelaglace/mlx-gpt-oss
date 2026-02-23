"""
MLX GPT-OSS 
OpenAI-compatible Pydantic models for chat completions.
Can be extended to more endpoints.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ChatCompletionRequest(BaseModel):
    """OpenAI ``/v1/chat/completions`` request body."""

    model: str
    messages: list[dict[str, Any]]
    seed: int | None = Field(None, ge=0)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, gt=0.0, le=1.0)
    top_k: int | None = Field(None, ge=0)
    max_tokens: int | None = Field(None, ge=1)
    repetition_penalty: float | None = Field(None, gt=0.0)
    repetition_context_size: int | None = Field(None, ge=1)
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    stop: str | list[str] | None = None


# ---------------------------------------------------------------------------
# Response models — non-streaming
# ---------------------------------------------------------------------------


class FunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


class ChatCompletionMessageToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall
    index: int | None = None


class Message(BaseModel):
    role: str = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    refusal: str | None = None
    tool_calls: list[ChatCompletionMessageToolCall] | None = None
    tool_call_id: str | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: UsageInfo | None = None
    request_id: str | None = None


# ---------------------------------------------------------------------------
# Response models — streaming (SSE chunks)
# ---------------------------------------------------------------------------


class ChoiceDeltaFunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


class ChoiceDeltaToolCall(BaseModel):
    index: int
    id: str | None = None
    type: str | None = "function"
    function: ChoiceDeltaFunctionCall | None = None


class Delta(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ChoiceDeltaToolCall] | None = None


class StreamingChoice(BaseModel):
    index: int = 0
    delta: Delta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamingChoice]
    usage: UsageInfo | None = None
    request_id: str | None = None


# ---------------------------------------------------------------------------
# Models & health
# ---------------------------------------------------------------------------


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "local"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[Model]


class HealthCheckStatus(BaseModel):
    status: Literal["ok", "unhealthy"]
    model_id: str | None = None
    active_requests: int = 0
    queued_requests: int = 0
