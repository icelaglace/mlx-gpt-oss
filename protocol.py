from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Literal


_TEXT_PART_TYPES = {"input_text", "text", "output_text"}
_UNSUPPORTED_PART_TYPES = {
    "input_image",
    "image",
    "image_url",
    "input_audio",
    "audio",
    "file",
}
_SUPPORTED_RESPONSE_KEYS = {
    "input",
    "instructions",
    "max_output_tokens",
    "metadata",
    "model",
    "parallel_tool_calls",
    "previous_response_id",
    "reasoning",
    "seed",
    "store",
    "stream",
    "temperature",
    "text",
    "tool_choice",
    "tools",
    "top_p",
}


class InvalidRequestError(ValueError):
    def __init__(self, message: str, *, param: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.param = param


@dataclass(frozen=True)
class NormalizedToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class NormalizedUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class NormalizedTurnResult:
    model: str
    content: str | None
    reasoning_content: str | None
    tool_calls: list[NormalizedToolCall]
    finish_reason: str
    usage: NormalizedUsage


@dataclass(frozen=True)
class NormalizedStreamEvent:
    kind: Literal[
        "text_delta",
        "reasoning_delta",
        "function_call",
        "completed",
        "failed",
    ]
    text: str | None = None
    tool_call: NormalizedToolCall | None = None
    result: NormalizedTurnResult | None = None
    error: str | None = None


@dataclass(frozen=True)
class NormalizedTurnRequest:
    model: str
    messages: list[dict[str, Any]]
    seed: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None
    repetition_context_size: int | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    reasoning_effort: str | None = None
    stop: str | list[str] | None = None
    metadata: dict[str, Any] | None = None
    store: bool | None = None
    previous_response_id: str | None = None
    instructions: str | None = None
    input_items: list[dict[str, Any]] = field(default_factory=list)
    base_transcript: list[dict[str, Any]] = field(default_factory=list)
    transcript_additions: list[dict[str, Any]] = field(default_factory=list)
    api_format: Literal["chat", "responses"] = "chat"


def _new_id(prefix: str) -> str:
    import random
    import time

    return f"{prefix}_{int(time.time())}{random.randint(0, 999999):06d}"


def _stringify_item(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _as_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    return None


def _normalize_response_text_parts(
    content: Any,
    *,
    param: str,
) -> tuple[str, list[dict[str, str]]]:
    if isinstance(content, str):
        return content, [{"type": "input_text", "text": content}]

    if not isinstance(content, list):
        raise InvalidRequestError(
            "Only text message content is supported for /v1/responses.",
            param=param,
        )

    texts: list[str] = []
    normalized_parts: list[dict[str, str]] = []
    for part_index, part in enumerate(content):
        if not isinstance(part, dict):
            raise InvalidRequestError(
                "Each input content part must be an object.",
                param=f"{param}[{part_index}]",
            )
        part_type = str(part.get("type") or "").strip()
        if part_type in _UNSUPPORTED_PART_TYPES:
            raise InvalidRequestError(
                f"Unsupported input content part type: {part_type}",
                param=f"{param}[{part_index}].type",
            )
        if part_type not in _TEXT_PART_TYPES:
            raise InvalidRequestError(
                f"Unsupported input content part type: {part_type or 'unknown'}",
                param=f"{param}[{part_index}].type",
            )
        text = part.get("text")
        if not isinstance(text, str):
            raise InvalidRequestError(
                "Text content parts must include a string `text` field.",
                param=f"{param}[{part_index}].text",
            )
        texts.append(text)
        normalized_parts.append({"type": part_type, "text": text})
    return "\n".join(texts), normalized_parts


def _normalize_response_input_item(
    item: dict[str, Any],
    *,
    index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    item_type = item.get("type")
    if item_type is None and ("role" in item or "content" in item):
        item_type = "message"

    if item_type == "message":
        role = str(item.get("role") or "user")
        text, normalized_content = _normalize_response_text_parts(
            item.get("content"),
            param=f"input[{index}].content",
        )
        input_item = {
            "id": item.get("id") or _new_id("item"),
            "type": "message",
            "role": role,
            "content": normalized_content,
        }
        conversation_message = {"role": role, "content": text}
        return input_item, conversation_message

    if item_type == "function_call_output":
        call_id = item.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise InvalidRequestError(
                "`function_call_output` items require `call_id`.",
                param=f"input[{index}].call_id",
            )
        output = _stringify_item(item.get("output"))
        input_item = {
            "id": item.get("id") or _new_id("item"),
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        }
        conversation_message = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": output,
        }
        return input_item, conversation_message

    if item_type == "function_call":
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise InvalidRequestError(
                "`function_call` items require `name`.",
                param=f"input[{index}].name",
            )
        call_id = item.get("call_id") or item.get("id") or _new_id("call")
        arguments = item.get("arguments", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        input_item = {
            "id": item.get("id") or _new_id("item"),
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        }
        conversation_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        }
        return input_item, conversation_message

    raise InvalidRequestError(
        f"Unsupported input item type: {item_type or 'unknown'}",
        param=f"input[{index}].type",
    )


def normalize_response_input(
    raw_input: str | list[Any] | dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if raw_input is None:
        raise InvalidRequestError("`input` is required.", param="input")

    if isinstance(raw_input, str):
        item_id = _new_id("item")
        return (
            [
                {
                    "id": item_id,
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": raw_input}],
                }
            ],
            [{"role": "user", "content": raw_input}],
        )

    items = [raw_input] if isinstance(raw_input, dict) else raw_input
    if not isinstance(items, list):
        raise InvalidRequestError(
            "`input` must be a string, object, or list.",
            param="input",
        )

    normalized_items: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise InvalidRequestError(
                "Each input item must be an object.",
                param=f"input[{index}]",
            )
        normalized_item, message = _normalize_response_input_item(item, index=index)
        normalized_items.append(normalized_item)
        messages.append(message)
    return normalized_items, messages


def normalize_response_tools(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if not tools:
        return None

    normalized: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise InvalidRequestError(
                "Each tool must be an object.",
                param=f"tools[{index}]",
            )
        tool_type = tool.get("type")
        if tool_type != "function":
            raise InvalidRequestError(
                f"Unsupported tool type: {tool_type}",
                param=f"tools[{index}].type",
            )
        tool_copy = copy.deepcopy(tool)
        function = tool_copy.get("function")
        if not isinstance(function, dict):
            # Compatibility: accept Responses-style function tool shape where
            # name/description/parameters live at the top level.
            top_level_name = tool_copy.get("name")
            if isinstance(top_level_name, str) and top_level_name:
                function = {"name": top_level_name}
                if isinstance(tool_copy.get("description"), str):
                    function["description"] = tool_copy["description"]
                if isinstance(tool_copy.get("parameters"), dict):
                    function["parameters"] = tool_copy["parameters"]
                if "strict" in tool_copy:
                    function["strict"] = tool_copy["strict"]
                tool_copy["function"] = function
            else:
                raise InvalidRequestError(
                    "Function tools require a `function` object or top-level function fields (`name`, `parameters`).",
                    param=f"tools[{index}].function",
                )

        top_level_name = tool_copy.get("name")
        if (
            not isinstance(function.get("name"), str) or not function.get("name")
        ) and isinstance(top_level_name, str) and top_level_name:
            function["name"] = top_level_name
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise InvalidRequestError(
                "Function tools require a non-empty `name`.",
                param=f"tools[{index}].function.name",
            )
        normalized.append(tool_copy)
    return normalized


def validate_response_request(payload: Any) -> None:
    extra = getattr(payload, "__pydantic_extra__", None) or {}
    for key in extra:
        if key not in _SUPPORTED_RESPONSE_KEYS:
            raise InvalidRequestError(
                f"Unsupported request field: {key}",
                param=key,
            )

    if payload.parallel_tool_calls is False:
        raise InvalidRequestError(
            "`parallel_tool_calls=false` is not supported.",
            param="parallel_tool_calls",
        )

    text_config = _as_dict(payload.text)
    if text_config and text_config.get("format") not in (None, {"type": "text"}):
        raise InvalidRequestError(
            "Only plain text responses are supported.",
            param="text.format",
        )

    if payload.tool_choice not in (None, "auto", "none"):
        raise InvalidRequestError(
            "Only `tool_choice` values `auto` and `none` are supported.",
            param="tool_choice",
        )

    reasoning_config = _as_dict(payload.reasoning)
    if payload.reasoning is not None and reasoning_config is None:
        raise InvalidRequestError(
            "`reasoning` must be an object when provided.",
            param="reasoning",
        )


def normalize_chat_request(payload: Any) -> NormalizedTurnRequest:
    tools = copy.deepcopy(payload.tools) if payload.tools else None
    tool_choice = payload.tool_choice
    if tool_choice == "none":
        tools = None

    return NormalizedTurnRequest(
        model=payload.model,
        messages=copy.deepcopy(payload.messages),
        seed=payload.seed,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
        repetition_penalty=payload.repetition_penalty,
        repetition_context_size=payload.repetition_context_size,
        stream=payload.stream,
        tools=tools,
        tool_choice=tool_choice,
        reasoning_effort=payload.reasoning_effort,
        stop=payload.stop,
        api_format="chat",
    )


def normalize_responses_request(
    payload: Any,
    *,
    previous_transcript: list[dict[str, Any]] | None = None,
) -> NormalizedTurnRequest:
    validate_response_request(payload)

    tools = normalize_response_tools(payload.tools)
    if payload.tool_choice == "none":
        tools = None

    input_items, input_messages = normalize_response_input(payload.input)
    messages: list[dict[str, Any]] = []
    if payload.instructions:
        messages.append({"role": "system", "content": payload.instructions})
    if previous_transcript:
        messages.extend(copy.deepcopy(previous_transcript))
    messages.extend(copy.deepcopy(input_messages))

    reasoning_effort = None
    reasoning_config = _as_dict(payload.reasoning)
    if isinstance(reasoning_config, dict):
        effort = reasoning_config.get("effort")
        if isinstance(effort, str):
            reasoning_effort = effort

    return NormalizedTurnRequest(
        model=payload.model,
        messages=messages,
        seed=payload.seed,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_output_tokens,
        stream=payload.stream,
        tools=tools,
        tool_choice=payload.tool_choice,
        reasoning_effort=reasoning_effort,
        metadata=copy.deepcopy(payload.metadata),
        store=True if payload.store is None else bool(payload.store),
        previous_response_id=payload.previous_response_id,
        instructions=payload.instructions,
        input_items=input_items,
        base_transcript=copy.deepcopy(previous_transcript or []),
        transcript_additions=copy.deepcopy(input_messages),
        api_format="responses",
    )


def build_assistant_transcript_message(
    result: NormalizedTurnResult,
) -> dict[str, Any]:
    if result.tool_calls:
        return {
            "role": "assistant",
            "content": result.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                }
                for tool_call in result.tool_calls
            ],
        }

    return {"role": "assistant", "content": result.content}


def build_response_output_items(
    result: NormalizedTurnResult,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if result.content is not None or not result.tool_calls:
        output.append(
            {
                "id": _new_id("msg"),
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": result.content or "",
                        "annotations": [],
                    }
                ],
            }
        )

    for tool_call in result.tool_calls:
        output.append(
            {
                "id": _new_id("fc"),
                "type": "function_call",
                "status": "completed",
                "call_id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            }
        )
    return output
