"""
MLX GPT-OSS 
OpenAI-compatible Pydantic models for chat completions.
Can be extended to more endpoints.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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
    response_format: dict[str, Any] | None = None
    stop: str | list[str] | None = None


class ResponsesReasoning(BaseModel):
    effort: Literal["low", "medium", "high"] | None = None


class ResponsesTextConfig(BaseModel):
    format: dict[str, Any] | None = None


class ResponsesCreateRequest(BaseModel):
    """OpenAI ``/v1/responses`` request body."""

    model_config = ConfigDict(extra="allow")

    model: str
    input: str | dict[str, Any] | list[Any]
    instructions: str | None = None
    stream: bool = False
    store: bool | None = None
    previous_response_id: str | None = None
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, gt=0.0, le=1.0)
    max_output_tokens: int | None = Field(None, ge=1)
    seed: int | None = Field(None, ge=0)
    metadata: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    reasoning: ResponsesReasoning | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    text: ResponsesTextConfig | dict[str, Any] | None = None


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


class ResponseOutputText(BaseModel):
    type: str = "output_text"
    text: str
    annotations: list[dict[str, Any]] = Field(default_factory=list)


class ResponseMessageItem(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    status: str = "completed"
    content: list[ResponseOutputText]


class ResponseFunctionCallItem(BaseModel):
    id: str
    type: str = "function_call"
    status: str = "completed"
    call_id: str
    name: str
    arguments: str


class ResponseUsageDetails(BaseModel):
    cached_tokens: int = 0
    reasoning_tokens: int = 0


class ResponseUsage(BaseModel):
    input_tokens: int
    input_tokens_details: ResponseUsageDetails
    output_tokens: int
    output_tokens_details: ResponseUsageDetails
    total_tokens: int


class ResponseObject(BaseModel):
    id: str
    object: str = "response"
    created_at: int
    status: str
    error: dict[str, Any] | None = None
    incomplete_details: dict[str, Any] | None = None
    instructions: str | None = None
    metadata: dict[str, Any] | None = None
    model: str
    output: list[dict[str, Any]]
    parallel_tool_calls: bool = True
    temperature: float | None = None
    tool_choice: str | dict[str, Any] | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    top_p: float | None = None
    max_output_tokens: int | None = None
    previous_response_id: str | None = None
    reasoning: dict[str, Any] | None = None
    text: dict[str, Any] | None = None
    usage: ResponseUsage | None = None


class ResponseDeleted(BaseModel):
    id: str
    object: str = "response.deleted"
    deleted: bool = True


class ResponseInputItemsList(BaseModel):
    object: str = "list"
    data: list[dict[str, Any]]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False
