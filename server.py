"""
MLX GPT-OSS Server —
Uses mlx-lm for inference, openai_harmony for prompt formatting, FastAPI for the HTTP layer.
Request queueing keeps /health responsive.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import Event
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from harmony import HarmonyParser, get_stop_tokens, render_harmony_prompt
from protocol import (
    InvalidRequestError,
    NormalizedStreamEvent,
    NormalizedToolCall,
    NormalizedTurnRequest,
    NormalizedTurnResult,
    NormalizedUsage,
    build_assistant_transcript_message,
    build_response_output_items,
    normalize_chat_request,
    normalize_responses_request,
)
from response_store import ResponseStore, StoredResponseRecord
from schemas import (
    ChatCompletionChunk,
    ChatCompletionMessageToolCall,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
    Delta,
    FunctionCall,
    HealthCheckStatus,
    Message,
    Model,
    ModelsResponse,
    ResponseDeleted,
    ResponseInputItemsList,
    ResponseObject,
    ResponseUsage,
    ResponseUsageDetails,
    ResponsesCreateRequest,
    StreamingChoice,
    UsageInfo,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_CURRENT_LOG_LEVEL = "INFO"
_CONSOLE_LOG_FORMAT = (
    "<green>{time:HH:mm:ss.SS}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)
_FILE_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SS} | {level: <8} | {message}"
_DEBUG_PREVIEW_LOG_FORMAT = "{}| rid=<cyan>{}</cyan> chars={} text=<dim>{}</dim>"
_METRICS_LOG_FORMAT = (
    "<cyan>{}</cyan>| rid=<cyan>{}</cyan> "
    "tok=<green>{}/{}</green> dur=<green>{:.2f}s</green> tps=<green>{:.2f}</green>"
)

_DEBUG_RAW_PREVIEW_CHARS = 0
_DEBUG_EVENT_WIDTH = 11

_HTTP_ACCESS_LOG_ENABLED = False

_STARTUP_DEBUG_ARGS = "args(unset)"


def configure_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """Configure console logging and optional rotating file logging."""
    global _CURRENT_LOG_LEVEL
    _CURRENT_LOG_LEVEL = log_level.upper()

    logger.remove()
    
    logger.level("DEBUG",    color="<fg #7f8c8d>")
    logger.level("INFO",     color="<bold>")
    logger.level("WARNING",  color="<bold><yellow>")
    logger.level("ERROR",    color="<bold><red>")

    logger.add(
        sys.stderr,
        format=_CONSOLE_LOG_FORMAT,
        level=_CURRENT_LOG_LEVEL,
        colorize=True,
        backtrace=True,
        diagnose=False,
    )

    if log_file:
        logger.add(
            log_file,
            format=_FILE_LOG_FORMAT,
            level=_CURRENT_LOG_LEVEL,
            backtrace=True,
            diagnose=False,
            rotation="25 MB",
            retention=5,
            enqueue=True,
        )


def _debug_raw_preview_enabled() -> bool:
    return _CURRENT_LOG_LEVEL == "DEBUG" and _DEBUG_RAW_PREVIEW_CHARS > 0


def _new_request_log_id() -> str:
    return f"{random.randint(0, 0xFFFF):04x}"


def _request_log_id(http_request: Request) -> str:
    request_log_id = getattr(http_request.state, "request_log_id", None)
    if isinstance(request_log_id, str) and request_log_id:
        return request_log_id
    request_log_id = _new_request_log_id()
    http_request.state.request_log_id = request_log_id
    return request_log_id


def _quote_preview(text: str) -> str:
    return json.dumps(text, ensure_ascii=True)


def _event_label(name: str) -> str:
    return f"{name:<{_DEBUG_EVENT_WIDTH}}"


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

# Just one is enough.
_executor = ThreadPoolExecutor(max_workers=1)


class HarmonyModel:
    """Wrapper around mlx-lm for GPT-OSS inference."""

    def __init__(self, model_path: str, context_length: int | None = None) -> None:
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler
        from mlx_lm.utils import load

        logger.opt(colors=True).info("Loading model: <magenta>{}</magenta>", model_path)
        
        self.model_path = model_path
        self.context_length = context_length
        self.model, self.tokenizer = load(model_path)
        self._make_sampler = make_sampler
        self._make_logits_processors = make_logits_processors
        self._stream_generate = stream_generate

        # Extend EOS tokens with Harmony stop tokens. Important for Harmony tool parsing.
        stop_tokens = get_stop_tokens()
        existing_eos = getattr(self.tokenizer, "eos_token_ids", [])
        
        if existing_eos is None:
            existing_eos = []
        elif isinstance(existing_eos, int):
            existing_eos = [existing_eos]
        else:
            existing_eos = list(existing_eos)
        for tid in stop_tokens:
            if tid not in existing_eos:
                existing_eos.append(tid)
        self.tokenizer.eos_token_ids = existing_eos

    def tokenize(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt)

    def generate_stream(
        self,
        input_ids: list[int],
        *,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.08,
        repetition_context_size: int = 128,
        seed: int | None = None,
    ):
        if seed is not None:
            self._set_seed(seed)
        sampler = self._make_sampler(temp=temperature, top_p=top_p, top_k=0)
        logits_processors = self._make_logits_processors(
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )
        generate_kwargs = {
            "prompt": input_ids,
            "max_tokens": max_tokens,
            "sampler": sampler,
            "logits_processors": logits_processors,
        }
        if self.context_length is not None:
            generate_kwargs["max_kv_size"] = self.context_length

        for response in self._stream_generate(
            self.model,
            self.tokenizer,
            **generate_kwargs,
        ):
            yield response

    def _set_seed(self, seed: int) -> None:
        import mlx.core as mx
        mx.random.seed(seed)


# ---------------------------------------------------------------------------
# Request queue  (keeps /health responsive during inference)
# ---------------------------------------------------------------------------


class RequestQueue:
    """
    Simple async request queue backed by a thread pool.
    Inference runs in a thread via ``run_in_executor``
    This way, we can call /health anytime.
    """

    def __init__(self) -> None:
        self._maxsize = 32
        self._queue: asyncio.Queue[
            tuple[asyncio.Future[Any], Any, tuple[Any, ...], dict[str, Any]]
        ] = asyncio.Queue(maxsize=self._maxsize)
        self._active = 0
        self._worker: asyncio.Task | None = None

    async def start(self) -> None:
        self._queue = asyncio.Queue(maxsize=self._maxsize)
        self._worker = asyncio.create_task(self._loop())
        logger.opt(colors=True).info( "Server is <green>ready</green>" )

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    def stats(self) -> dict[str, int]:
        return {"active_requests": self._active, "queued_requests": self._queue.qsize()}

    async def submit(self, coro_func, *args, **kwargs):
        """Submit work and wait for the result."""
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put((future, coro_func, args, kwargs))
        return await future

    async def _loop(self) -> None:
        while True:
            future, coro_func, args, kwargs = await self._queue.get()
            self._active += 1
            try:
                result = await coro_func(*args, **kwargs)
            except Exception as exc:
                logger.exception("Request worker failed")
                if not future.done():
                    try:
                        future.set_exception(exc)
                    except asyncio.InvalidStateError:
                        logger.debug(
                            "Skipped exception delivery: request future is no longer pending"
                        )
            else:
                if not future.done():
                    try:
                        future.set_result(result)
                    except asyncio.InvalidStateError:
                        logger.debug(
                            "Skipped result delivery: request future is no longer pending"
                        )
            finally:
                self._active -= 1


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

# GPT-OSS defaults (OpenAI recommendations).
_DEFAULT_TEMP = 1.0
_DEFAULT_TOP_P = 1.0
_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_REPETITION_PENALTY = 1.08
_DEFAULT_REPETITION_CONTEXT_SIZE = 128
_ALLOWED_REASONING_EFFORTS = {"low", "medium", "high"}
_FINISH_REASON_MAP = {
    "max_tokens": "length",
    "length": "length",
    "stop": "stop",
    "eos": "stop",
}


def _new_prefixed_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time())}{random.randint(0, 999999):06d}"


def _sanitize_tool_name(name: str | None) -> str | None:
    if not name:
        return None
    candidate = name.strip()
    marker_idx = candidate.find("<|")
    if marker_idx != -1:
        candidate = candidate[:marker_idx]
    candidate = candidate.strip().strip(" \t\r\n.,:;\"'")
    match = re.match(r"[A-Za-z_][A-Za-z0-9_.-]*", candidate)
    if not match:
        return None
    return match.group(0)


def _normalize_tool_call(
    tool_call: dict[str, Any],
    *,
    warning_label: str = "tool call",
) -> tuple[str, str] | None:
    """Return sanitized ``(name, arguments_json_or_string)`` for tool calls."""
    sanitized_name = _sanitize_tool_name(tool_call.get("name"))
    if not sanitized_name:
        logger.warning(
            "Dropping malformed {} name={!r}",
            warning_label,
            tool_call.get("name"),
        )
        return None

    args = tool_call.get("arguments", "")
    if not isinstance(args, str):
        args = json.dumps(args)
    return sanitized_name, args


def _common_prefix_len(left: str, right: str) -> int:
    max_len = min(len(left), len(right))
    idx = 0
    while idx < max_len and left[idx] == right[idx]:
        idx += 1
    return idx


def _only_unseen_suffix(full_text: str, already_emitted: str) -> str:
    if not full_text:
        return ""
    if not already_emitted:
        return full_text
    return full_text[_common_prefix_len(full_text, already_emitted) :]


def _drop_emitted_tool_call_prefix(
    tool_calls: list[dict[str, Any]],
    emitted_tool_calls: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """
    Drop a matching emitted prefix while preserving valid repeated tool calls.

    During buffered fallback, parser output can include tool calls that were
    already streamed. We only trim an exact prefix match to avoid replaying
    those earlier calls.
    """
    if not tool_calls or not emitted_tool_calls:
        return tool_calls

    max_compare = min(len(tool_calls), len(emitted_tool_calls))
    drop_count = 0
    for idx in range(max_compare):
        normalized = _normalize_tool_call(
            tool_calls[idx],
            warning_label="stream fallback tool call",
        )
        if normalized is None or normalized != emitted_tool_calls[idx]:
            break
        drop_count += 1
    return tool_calls[drop_count:]


def _normalize_reasoning_effort(value: str | None) -> str:
    if value is None:
        return "medium"

    normalized = value.strip().lower()
    if normalized in _ALLOWED_REASONING_EFFORTS:
        return normalized

    logger.warning(
        "Invalid reasoning_effort={!r}; falling back to 'medium'.",
        value,
    )
    return "medium"


def _normalize_stop_sequences(stop: str | list[str] | None) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    return [item for item in stop if item]


def _find_first_stop_index(text: str, stop_sequences: list[str]) -> int | None:
    first_index: int | None = None
    for stop_sequence in stop_sequences:
        idx = text.find(stop_sequence)
        if idx == -1:
            continue
        if first_index is None or idx < first_index:
            first_index = idx
    return first_index


def _truncate_text_at_stop(
    text: str | None, stop_sequences: list[str]
) -> tuple[str | None, bool]:
    if text is None or not stop_sequences:
        return text, False
    stop_idx = _find_first_stop_index(text, stop_sequences)
    if stop_idx is None:
        return text, False
    return text[:stop_idx], True


def _process_stop_sequences_chunk(
    chunk_text: str,
    stop_sequences: list[str],
    tail: str,
    max_stop_len: int,
) -> tuple[str, str, bool]:
    """
    Process a streaming text chunk against stop sequences.
    Returns ``(emit_text, new_tail, stop_found)`` where:
    - ``emit_text`` is safe to emit now,
    - ``new_tail`` must be retained for cross-chunk stop matching,
    - ``stop_found`` indicates stop sequence matched and stream should stop.
    """
    combined = tail + chunk_text
    stop_idx = _find_first_stop_index(combined, stop_sequences)
    if stop_idx is not None:
        return combined[:stop_idx], "", True

    if max_stop_len <= 1:
        return combined, "", False

    tail_len = max_stop_len - 1
    if len(combined) <= tail_len:
        return "", combined, False

    return combined[:-tail_len], combined[-tail_len:], False


def _normalize_finish_reason(finish_reason: Any) -> str:
    normalized = "stop" if finish_reason is None else str(finish_reason).strip().lower()
    return _FINISH_REASON_MAP.get(normalized, "stop")


def _resolve_sampling_params(
    request: Any,
) -> tuple[float, float, int, float, int]:
    temperature = (
        request.temperature if request.temperature is not None else _DEFAULT_TEMP
    )
    top_p = request.top_p if request.top_p is not None else _DEFAULT_TOP_P
    repetition_penalty = (
        request.repetition_penalty
        if request.repetition_penalty is not None
        else _DEFAULT_REPETITION_PENALTY
    )
    repetition_context_size = (
        request.repetition_context_size
        if request.repetition_context_size is not None
        else _DEFAULT_REPETITION_CONTEXT_SIZE
    )

    max_tokens = request.max_tokens or _DEFAULT_MAX_TOKENS
    if (
        _model is not None
        and _model.context_length is not None
        and max_tokens > _model.context_length
    ):
        logger.warning(
            "Capping client max_tokens={} to model context_length={} to avoid invalid requests.",
            max_tokens,
            _model.context_length,
        )
        max_tokens = _model.context_length

    return (
        temperature,
        top_p,
        max_tokens,
        repetition_penalty,
        repetition_context_size,
    )


def _response_model_id(request_model: str) -> str:
    if _model is not None:
        model_path = getattr(_model, "model_path", None)
        if isinstance(model_path, str) and model_path:
            return model_path
    return request_model


def _prepare_prompt(
    request: Any,
    *,
    request_log_id: str | None = None,
) -> list[int]:
    """Build input_ids from the request messages."""
    reasoning_effort = _normalize_reasoning_effort(
        getattr(request, "reasoning_effort", None)
    )

    prompt_str = render_harmony_prompt(
        messages=request.messages,
        tools=request.tools,
        reasoning_effort=reasoning_effort,
    )
    if _debug_raw_preview_enabled():
        prompt_preview = prompt_str[:_DEBUG_RAW_PREVIEW_CHARS]
        logger.opt(colors=True).debug(
            _DEBUG_PREVIEW_LOG_FORMAT,
            _event_label("PROMPT"),
            request_log_id or "-",
            len(prompt_preview),
            _quote_preview(prompt_preview),
        )

    return _model.tokenize(prompt_str)


def _build_normalized_tool_calls(
    tool_calls: list[dict[str, Any]] | None,
) -> list[NormalizedToolCall]:
    normalized_tool_calls: list[NormalizedToolCall] = []
    for tool_call in tool_calls or []:
        normalized_tool_call = _normalize_tool_call(tool_call)
        if normalized_tool_call is None:
            continue
        tool_name, arguments = normalized_tool_call
        normalized_tool_calls.append(
            NormalizedToolCall(
                id=_new_prefixed_id("call"),
                name=tool_name,
                arguments=arguments,
            )
        )
    return normalized_tool_calls


def _turn_result_from_parsed(
    parsed: dict[str, Any],
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    finish_reason: str = "stop",
    tool_calls: list[NormalizedToolCall] | None = None,
) -> NormalizedTurnResult:
    resolved_tool_calls = (
        list(tool_calls) if tool_calls is not None else _build_normalized_tool_calls(parsed.get("tool_calls"))
    )
    resolved_finish_reason = "tool_calls" if resolved_tool_calls else finish_reason
    usage = NormalizedUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return NormalizedTurnResult(
        model=model,
        content=parsed.get("content"),
        reasoning_content=parsed.get("reasoning_content"),
        tool_calls=resolved_tool_calls,
        finish_reason=resolved_finish_reason,
        usage=usage,
    )


def _format_chat_completion_from_result(
    result: NormalizedTurnResult,
) -> ChatCompletionResponse:
    tc_objects = [
        ChatCompletionMessageToolCall(
            id=tool_call.id,
            function=FunctionCall(
                name=tool_call.name,
                arguments=tool_call.arguments,
            ),
            index=index,
        )
        for index, tool_call in enumerate(result.tool_calls)
    ]
    return ChatCompletionResponse(
        id=_new_prefixed_id("chatcmpl"),
        created=int(time.time()),
        model=result.model,
        choices=[
            Choice(
                message=Message(
                    role="assistant",
                    content=result.content,
                    reasoning_content=result.reasoning_content,
                    tool_calls=tc_objects or None,
                ),
                finish_reason=result.finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
        ),
    )


async def _generate_turn_result(
    request: NormalizedTurnRequest,
    *,
    request_log_id: str | None = None,
) -> NormalizedTurnResult:
    """Non-streaming generation — runs inference in a thread."""
    log_id = request_log_id or _new_request_log_id()
    request_start_ts = time.time()
    input_ids = _prepare_prompt(request, request_log_id=log_id)
    prompt_len = len(input_ids)

    (
        temp,
        top_p,
        max_tokens,
        repetition_penalty,
        repetition_context_size,
    ) = _resolve_sampling_params(request)
    stop_sequences = _normalize_stop_sequences(request.stop)
    seed = request.seed

    loop = asyncio.get_running_loop()

    # Collect detokenized text from the sync generator in a thread.
    def _run():
        generated_parts: list[str] = []
        completion_tokens = 0
        model_finish_reason = "stop"
        for resp in _model.generate_stream(
            input_ids,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            seed=seed,
        ):
            if resp.finish_reason is None:
                completion_tokens += 1
            else:
                model_finish_reason = _normalize_finish_reason(resp.finish_reason)
            if resp.text:
                generated_parts.append(resp.text)
        return "".join(generated_parts), completion_tokens, model_finish_reason

    full_text, completion_tokens, model_finish_reason = await loop.run_in_executor(
        _executor, _run
    )

    if _debug_raw_preview_enabled():
        preview = full_text[:_DEBUG_RAW_PREVIEW_CHARS]
        logger.opt(colors=True).debug(
            _DEBUG_PREVIEW_LOG_FORMAT,
            _event_label("RAW_OUTPUT"),
            log_id,
            len(preview),
            _quote_preview(preview),
        )

    # Parse through Harmony.
    parser = HarmonyParser()
    parsed = parser.parse(full_text)
    parsed["content"], _ = _truncate_text_at_stop(parsed.get("content"), stop_sequences)

    result = _turn_result_from_parsed(
        parsed,
        _response_model_id(request.model),
        prompt_len,
        completion_tokens,
        finish_reason=model_finish_reason,
    )

    request_end_ts = time.time()
    elapsed_s = max(request_end_ts - request_start_ts, 1e-9)
    logger.opt(colors=True).debug(
        _METRICS_LOG_FORMAT,
        _event_label("METRICS"),
        log_id,
        prompt_len,
        completion_tokens,
        elapsed_s,
        completion_tokens / elapsed_s if completion_tokens else 0.0,
    )

    return result


async def _generate_response(
    request: ChatCompletionRequest,
    *,
    request_log_id: str | None = None,
) -> ChatCompletionResponse:
    result = await _generate_turn_result(
        normalize_chat_request(request),
        request_log_id=request_log_id,
    )
    return _format_chat_completion_from_result(result)


def _format_response(
    parsed: dict[str, Any],
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    finish_reason: str = "stop",
) -> ChatCompletionResponse:
    """Format parsed result into an OpenAI chat completion response."""
    return _format_chat_completion_from_result(
        _turn_result_from_parsed(
            parsed,
            model,
            prompt_tokens,
            completion_tokens,
            finish_reason=finish_reason,
        )
    )


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _sse(data: dict | ChatCompletionChunk) -> str:
    payload = data.model_dump() if isinstance(data, ChatCompletionChunk) else data
    return f"data: {json.dumps(payload)}\n\n"


@dataclass(frozen=True)
class _StreamRequestContext:
    """Immutable per-request inputs for the streaming code path."""

    log_id: str
    request_start_ts: float
    chat_id: str
    created: int
    model: str
    input_ids: list[int]
    prompt_len: int
    temp: float
    top_p: float
    max_tokens: int
    repetition_penalty: float
    repetition_context_size: int
    stop_sequences: list[str]
    seed: int | None
    max_stop_len: int
    parse_tools: bool


def _stream_chunk(
    *,
    chat_id: str,
    created: int,
    model: str,
    delta: Delta,
    finish_reason: str | None = None,
    usage: UsageInfo | None = None,
) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id=chat_id,
        created=created,
        model=model,
        choices=[StreamingChoice(delta=delta, finish_reason=finish_reason)],
        usage=usage,
    )


@dataclass
class _StreamState:
    """Mutable fields that evolve while a stream is running."""

    finish_reason: str = "stop"
    completion_tokens: int = 0
    stop_tail: str = ""
    raw_output_parts: list[str] = field(default_factory=list)
    emitted_content_parts: list[str] = field(default_factory=list)
    emitted_reasoning_parts: list[str] = field(default_factory=list)
    emitted_tool_calls: list[NormalizedToolCall] = field(default_factory=list)
    raw_preview_parts: list[str] = field(default_factory=list)
    raw_preview_len: int = 0
    raw_text_fallback: bool = False
    disconnected: bool = False
    stream_cancel_reason: str | None = None


def _build_stream_request_context(
    request: Any,
    *,
    request_log_id: str | None = None,
) -> _StreamRequestContext:
    log_id = request_log_id or _new_request_log_id()
    input_ids = _prepare_prompt(request, request_log_id=log_id)
    (
        temp,
        top_p,
        max_tokens,
        repetition_penalty,
        repetition_context_size,
    ) = _resolve_sampling_params(request)
    stop_sequences = _normalize_stop_sequences(request.stop)
    return _StreamRequestContext(
        log_id=log_id,
        request_start_ts=time.time(),
        chat_id=_new_prefixed_id("chatcmpl"),
        created=int(time.time()),
        model=_response_model_id(request.model),
        input_ids=input_ids,
        prompt_len=len(input_ids),
        temp=temp,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        stop_sequences=stop_sequences,
        seed=request.seed,
        max_stop_len=max((len(s) for s in stop_sequences), default=0),
        parse_tools=bool(request.tools),
    )


def _start_stream_worker(
    *,
    context: _StreamRequestContext,
    loop: asyncio.AbstractEventLoop,
    chunk_queue: asyncio.Queue[tuple[str, str | None, int] | None],
    stop_event: Event,
    worker_failed: Event,
) -> None:
    """Run model streaming in executor and push chunks into ``chunk_queue``."""

    def _run():
        try:
            for resp in _model.generate_stream(
                context.input_ids,
                max_tokens=context.max_tokens,
                temperature=context.temp,
                top_p=context.top_p,
                repetition_penalty=context.repetition_penalty,
                repetition_context_size=context.repetition_context_size,
                seed=context.seed,
            ):
                if stop_event.is_set():
                    break
                loop.call_soon_threadsafe(
                    chunk_queue.put_nowait,
                    (
                        resp.text or "",
                        (
                            _normalize_finish_reason(resp.finish_reason)
                            if resp.finish_reason is not None
                            else None
                        ),
                        int(getattr(resp, "generation_tokens", 0) or 0),
                    ),
                )
                if resp.finish_reason is not None:
                    break
        except Exception:
            worker_failed.set()
            logger.exception("Streaming worker failed")
        finally:
            try:
                loop.call_soon_threadsafe(chunk_queue.put_nowait, None)
            except RuntimeError:
                # Event loop can already be closed during server shutdown.
                pass

    _executor.submit(_run)


def _parse_streaming_fallback_output(
    raw_text: str,
    *,
    parse_tools: bool,
) -> dict[str, Any]:
    """Recover a structured response from raw streamed text after parser failure."""
    parser = HarmonyParser()
    parsed = parser.parse(raw_text)

    if not parse_tools:
        parsed["tool_calls"] = []
    return parsed


async def _stream_response(
    request: ChatCompletionRequest,
    client_request: Request | None = None,
    *,
    request_log_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """SSE streaming generator matching the OpenAI chat protocol."""
    normalized_request = normalize_chat_request(request)
    chat_id = _new_prefixed_id("chatcmpl")
    created = int(time.time())
    model = _response_model_id(normalized_request.model)
    tool_call_index = -1
    saw_terminal_event = False

    yield _sse(
        _stream_chunk(
            chat_id=chat_id,
            created=created,
            model=model,
            delta=Delta(role="assistant"),
        )
    )

    async for event in _stream_events(
        normalized_request,
        client_request=client_request,
        request_log_id=request_log_id,
    ):
        if event.kind == "text_delta" and event.text:
            yield _sse(
                _stream_chunk(
                    chat_id=chat_id,
                    created=created,
                    model=model,
                    delta=Delta(content=event.text),
                )
            )
            continue

        if event.kind == "reasoning_delta" and event.text:
            yield _sse(
                _stream_chunk(
                    chat_id=chat_id,
                    created=created,
                    model=model,
                    delta=Delta(reasoning_content=event.text),
                )
            )
            continue

        if event.kind == "function_call" and event.tool_call is not None:
            tool_call_index += 1
            yield _sse(
                _stream_chunk(
                    chat_id=chat_id,
                    created=created,
                    model=model,
                    delta=Delta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=tool_call_index,
                                id=event.tool_call.id,
                                function=ChoiceDeltaFunctionCall(
                                    name=event.tool_call.name,
                                    arguments=event.tool_call.arguments,
                                ),
                            )
                        ]
                    ),
                )
            )
            continue

        if event.kind in {"completed", "failed"} and event.result is not None:
            saw_terminal_event = True
            usage = UsageInfo(
                prompt_tokens=event.result.usage.prompt_tokens,
                completion_tokens=event.result.usage.completion_tokens,
                total_tokens=event.result.usage.total_tokens,
            )
            yield _sse(
                _stream_chunk(
                    chat_id=chat_id,
                    created=created,
                    model=model,
                    delta=Delta(),
                    finish_reason=event.result.finish_reason,
                    usage=usage,
                )
            )
            break

    if not saw_terminal_event:
        return

    yield "data: [DONE]\n\n"


def _stream_state_tool_pairs(state: _StreamState) -> list[tuple[str, str]]:
    return [(tool_call.name, tool_call.arguments) for tool_call in state.emitted_tool_calls]


def _build_stream_tool_call_events(
    tool_calls: list[dict[str, Any]],
    *,
    state: _StreamState,
) -> list[NormalizedStreamEvent]:
    pending_tool_calls = _drop_emitted_tool_call_prefix(
        tool_calls,
        _stream_state_tool_pairs(state),
    )
    events: list[NormalizedStreamEvent] = []
    for tool_call in pending_tool_calls:
        normalized_tool_call = _normalize_tool_call(
            tool_call,
            warning_label="stream tool call",
        )
        if normalized_tool_call is None:
            continue
        tool_name, arguments = normalized_tool_call
        normalized_event_tool_call = NormalizedToolCall(
            id=_new_prefixed_id("call"),
            name=tool_name,
            arguments=arguments,
        )
        state.emitted_tool_calls.append(normalized_event_tool_call)
        events.append(
            NormalizedStreamEvent(
                kind="function_call",
                tool_call=normalized_event_tool_call,
            )
        )
    if events:
        state.finish_reason = "tool_calls"
    return events


def _should_reparse_stream_output(
    raw_text: str,
    *,
    raw_text_fallback: bool,
) -> bool:
    if not raw_text:
        return False
    if raw_text_fallback:
        return True
    if "to=functions." not in raw_text:
        return False
    if not any(marker in raw_text for marker in ("<|channel|>", "<|start|>", "<|call|>")):
        return False
    return any(marker in raw_text for marker in ("<|message|>", "<|constrain|>", "<|call|>"))


def _should_parse_tool_calls_from_raw_stream_output(
    raw_text: str,
    *,
    parse_tools: bool,
) -> bool:
    if not parse_tools or "to=functions." not in raw_text:
        return False
    if not any(marker in raw_text for marker in ("<|channel|>", "<|start|>", "<|call|>")):
        return False
    return any(marker in raw_text for marker in ("<|message|>", "<|constrain|>", "<|call|>"))


def _build_reparsed_stream_events(
    raw_text: str,
    *,
    parse_tools: bool,
    stop_sequences: list[str],
    state: _StreamState,
) -> list[NormalizedStreamEvent]:
    parsed = _parse_streaming_fallback_output(
        raw_text,
        parse_tools=_should_parse_tool_calls_from_raw_stream_output(
            raw_text,
            parse_tools=parse_tools,
        ),
    )
    events: list[NormalizedStreamEvent] = []

    reasoning_content = parsed.get("reasoning_content")
    if reasoning_content:
        reasoning_content = _only_unseen_suffix(
            reasoning_content,
            "".join(state.emitted_reasoning_parts),
        )
    if reasoning_content:
        state.emitted_reasoning_parts.append(reasoning_content)
        events.append(
            NormalizedStreamEvent(
                kind="reasoning_delta",
                text=reasoning_content,
            )
        )

    parsed_content = parsed.get("content")
    if parsed_content:
        parsed_content, _ = _truncate_text_at_stop(
            parsed_content,
            stop_sequences,
        )
        parsed_content = _only_unseen_suffix(
            parsed_content,
            "".join(state.emitted_content_parts),
        )
    if parsed_content:
        state.emitted_content_parts.append(parsed_content)
        events.append(
            NormalizedStreamEvent(
                kind="text_delta",
                text=parsed_content,
            )
        )

    if parse_tools and parsed.get("tool_calls"):
        events.extend(
            _build_stream_tool_call_events(
                parsed["tool_calls"],
                state=state,
            )
        )

    return events


def _build_stream_result(
    context: _StreamRequestContext,
    state: _StreamState,
) -> NormalizedTurnResult:
    return NormalizedTurnResult(
        model=context.model,
        content="".join(state.emitted_content_parts) or None,
        reasoning_content="".join(state.emitted_reasoning_parts) or None,
        tool_calls=list(state.emitted_tool_calls),
        finish_reason=state.finish_reason,
        usage=NormalizedUsage(
            prompt_tokens=context.prompt_len,
            completion_tokens=state.completion_tokens,
            total_tokens=context.prompt_len + state.completion_tokens,
        ),
    )


async def _stream_events(
    request: NormalizedTurnRequest,
    client_request: Request | None = None,
    *,
    request_log_id: str | None = None,
) -> AsyncGenerator[NormalizedStreamEvent, None]:
    """Internal streaming generator yielding transport-neutral events."""
    global _stream_active_requests
    context = _build_stream_request_context(
        request,
        request_log_id=request_log_id,
    )
    parse_tools = context.parse_tools
    stop_sequences = context.stop_sequences
    state = _StreamState()
    stop_event = Event()
    worker_failed = Event()
    _stream_active_requests += 1

    # Run inference in background thread, feed chunks through an async queue.
    chunk_queue: asyncio.Queue[tuple[str, str | None, int] | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    _start_stream_worker(
        context=context,
        loop=loop,
        chunk_queue=chunk_queue,
        stop_event=stop_event,
        worker_failed=worker_failed,
    )

    parser = HarmonyParser()

    async def _client_disconnected() -> bool:
        if client_request is None:
            return False
        try:
            return await client_request.is_disconnected()
        except RuntimeError:
            # Request context may be broken during shutdown.
            return False

    try:
        while True:
            if await _client_disconnected():
                state.disconnected = True
                state.stream_cancel_reason = "disconnect_poll"
                break

            try:
                chunk = await asyncio.wait_for(chunk_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue

            if chunk is None:
                if worker_failed.is_set():
                    state.stream_cancel_reason = "worker_error"
                if stop_sequences and state.stop_tail:
                    state.emitted_content_parts.append(state.stop_tail)
                    yield NormalizedStreamEvent(
                        kind="text_delta",
                        text=state.stop_tail,
                    )
                    state.stop_tail = ""

                # End of stream so flush parser.
                if not state.raw_text_fallback and parse_tools:
                    try:
                        flushed, _ = parser.handle_stream_end(parse_tools=True)
                    except Exception as exc:
                        logger.warning(
                            "Harmony parse failed during stream flush. Switching to buffered fallback. error_type={}",
                            type(exc).__name__,
                        )
                        state.raw_text_fallback = True
                        flushed = None
                    if flushed and flushed.get("tool_calls"):
                        for event in _build_stream_tool_call_events(
                            flushed["tool_calls"],
                            state=state,
                        ):
                            yield event

                raw_text = "".join(state.raw_output_parts)
                if _should_reparse_stream_output(
                    raw_text,
                    raw_text_fallback=state.raw_text_fallback,
                ):
                    for event in _build_reparsed_stream_events(
                        raw_text,
                        parse_tools=parse_tools,
                        stop_sequences=stop_sequences,
                        state=state,
                    ):
                        yield event
                break

            token_text, model_finish_reason, generated_count = chunk
            state.completion_tokens = max(state.completion_tokens, generated_count)
            if model_finish_reason and state.finish_reason != "tool_calls":
                state.finish_reason = model_finish_reason
            if (
                _debug_raw_preview_enabled()
                and token_text
                and state.raw_preview_len < _DEBUG_RAW_PREVIEW_CHARS
            ):
                remaining = _DEBUG_RAW_PREVIEW_CHARS - state.raw_preview_len
                raw_piece = token_text[:remaining]
                state.raw_preview_parts.append(raw_piece)
                state.raw_preview_len += len(raw_piece)

            if not token_text:
                continue

            state.raw_output_parts.append(token_text)

            if state.raw_text_fallback:
                continue

            # Parse this chunk.
            try:
                parsed, is_complete = parser.parse_streaming(
                    token_text,
                    parse_tools=parse_tools,
                )
            except Exception as exc:
                logger.warning(
                    "Harmony parse failed (streaming). Switching to buffered fallback. error_type={}",
                    type(exc).__name__,
                )
                state.raw_text_fallback = True
                continue

            if parsed is None:
                continue

            # Emit reasoning content delta.
            reasoning_content = parsed.get("reasoning_content")
            if reasoning_content:
                state.emitted_reasoning_parts.append(reasoning_content)
                yield NormalizedStreamEvent(
                    kind="reasoning_delta",
                    text=reasoning_content,
                )

            # Emit content delta.
            content_delta = parsed.get("content")
            if content_delta:
                if stop_sequences:
                    emit_text, state.stop_tail, stop_found = (
                        _process_stop_sequences_chunk(
                            content_delta,
                            stop_sequences,
                            state.stop_tail,
                            context.max_stop_len,
                        )
                    )
                    if stop_found:
                        stop_event.set()
                        state.finish_reason = "stop"
                        if emit_text:
                            state.emitted_content_parts.append(emit_text)
                            yield NormalizedStreamEvent(
                                kind="text_delta",
                                text=emit_text,
                            )
                        break
                    content_delta = emit_text
                if content_delta:
                    state.emitted_content_parts.append(content_delta)
                    yield NormalizedStreamEvent(
                        kind="text_delta",
                        text=content_delta,
                    )

            # Emit tool call deltas.
            if parse_tools and parsed.get("tool_calls"):
                for event in _build_stream_tool_call_events(
                    parsed["tool_calls"],
                    state=state,
                ):
                    yield event

            if is_complete:
                break
    except asyncio.CancelledError:
        state.stream_cancel_reason = "asgi_cancelled"
        raise
    finally:
        stop_event.set()
        _stream_active_requests = max(0, _stream_active_requests - 1)
    if state.stream_cancel_reason:
        logger.opt(colors=True).debug(
            "<red>{}</red>| rid=<cyan>{}</cyan> reason={}",
            _event_label("STREAM_CANCEL"),
            context.log_id,
            state.stream_cancel_reason,
        )

    if state.disconnected:
        return

    if _debug_raw_preview_enabled():
        raw_preview = "".join(state.raw_preview_parts)
        logger.opt(colors=True).debug(
            _DEBUG_PREVIEW_LOG_FORMAT,
            _event_label("RAW_OUTPUT"),
            context.log_id,
            len(raw_preview),
            _quote_preview(raw_preview),
        )

    request_end_ts = time.time()
    elapsed_s = max(request_end_ts - context.request_start_ts, 1e-9)
    logger.opt(colors=True).debug(
        _METRICS_LOG_FORMAT,
        _event_label("METRICS"),
        context.log_id,
        context.prompt_len,
        state.completion_tokens,
        elapsed_s,
        state.completion_tokens / elapsed_s if state.completion_tokens else 0.0,
    )
    result = _build_stream_result(context, state)
    if worker_failed.is_set():
        yield NormalizedStreamEvent(
            kind="failed",
            result=result,
            error="Streaming worker failed",
        )
        return
    yield NormalizedStreamEvent(kind="completed", result=result)


# ---------------------------------------------------------------------------
# Responses helpers
# ---------------------------------------------------------------------------


def _invalid_request_response(message: str, *, param: str | None = None) -> JSONResponse:
    error: dict[str, Any] = {
        "message": message,
        "type": "invalid_request_error",
    }
    if param is not None:
        error["param"] = param
    return JSONResponse(status_code=400, content={"error": error})


def _not_found_response(message: str, *, param: str | None = None) -> JSONResponse:
    error: dict[str, Any] = {
        "message": message,
        "type": "not_found_error",
    }
    if param is not None:
        error["param"] = param
    return JSONResponse(status_code=404, content={"error": error})


def _responses_usage_payload(result: NormalizedTurnResult) -> ResponseUsage:
    return ResponseUsage(
        input_tokens=result.usage.prompt_tokens,
        input_tokens_details=ResponseUsageDetails(),
        output_tokens=result.usage.completion_tokens,
        output_tokens_details=ResponseUsageDetails(),
        total_tokens=result.usage.total_tokens,
    )


def _responses_reasoning_payload(
    request: NormalizedTurnRequest,
) -> dict[str, Any] | None:
    if not request.reasoning_effort:
        return None
    return {"effort": _normalize_reasoning_effort(request.reasoning_effort)}


def _build_response_object(
    *,
    response_id: str,
    created_at: int,
    request: NormalizedTurnRequest,
    result: NormalizedTurnResult,
    status: str = "completed",
    output: list[dict[str, Any]] | None = None,
    error: dict[str, Any] | None = None,
) -> ResponseObject:
    return ResponseObject(
        id=response_id,
        created_at=created_at,
        status=status,
        error=error,
        incomplete_details=None,
        instructions=request.instructions,
        metadata=request.metadata,
        model=result.model,
        output=output if output is not None else build_response_output_items(result),
        parallel_tool_calls=True,
        temperature=request.temperature,
        tool_choice=request.tool_choice or "auto",
        tools=request.tools or [],
        top_p=request.top_p,
        max_output_tokens=request.max_tokens,
        previous_response_id=request.previous_response_id,
        reasoning=_responses_reasoning_payload(request),
        text={"format": {"type": "text"}},
        usage=_responses_usage_payload(result),
    )


def _record_request_options(request: NormalizedTurnRequest) -> dict[str, Any]:
    return {
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_output_tokens": request.max_tokens,
        "tool_choice": request.tool_choice or "auto",
        "tools": request.tools or [],
        "reasoning": _responses_reasoning_payload(request),
    }


def _store_response_result(
    *,
    response_id: str,
    created_at: int,
    request: NormalizedTurnRequest,
    result: NormalizedTurnResult,
    output: list[dict[str, Any]],
) -> None:
    if not request.store:
        return

    transcript = list(request.base_transcript)
    transcript.extend(request.transcript_additions)
    transcript.append(build_assistant_transcript_message(result))
    _response_store.put(
        StoredResponseRecord(
            id=response_id,
            created_at=created_at,
            model=result.model,
            metadata=request.metadata,
            request_options=_record_request_options(request),
            instructions=request.instructions,
            input_items=request.input_items,
            previous_response_id=request.previous_response_id,
            transcript=transcript,
            output=output,
            usage=_responses_usage_payload(result).model_dump(),
            status="completed",
            store=bool(request.store),
        )
    )


def _response_object_from_record(record: StoredResponseRecord) -> ResponseObject:
    options = record.request_options
    return ResponseObject(
        id=record.id,
        created_at=record.created_at,
        status=record.status,
        error=None,
        incomplete_details=None,
        instructions=record.instructions,
        metadata=record.metadata,
        model=record.model,
        output=record.output,
        parallel_tool_calls=True,
        temperature=options.get("temperature"),
        tool_choice=options.get("tool_choice") or "auto",
        tools=options.get("tools") or [],
        top_p=options.get("top_p"),
        max_output_tokens=options.get("max_output_tokens"),
        previous_response_id=record.previous_response_id,
        reasoning=options.get("reasoning"),
        text={"format": {"type": "text"}},
        usage=ResponseUsage(**record.usage),
    )


def _load_previous_transcript(previous_response_id: str | None) -> list[dict[str, Any]] | None:
    if not previous_response_id:
        return None
    record = _response_store.get(previous_response_id)
    if record is None:
        raise InvalidRequestError(
            f"Previous response `{previous_response_id}` was not found.",
            param="previous_response_id",
        )
    return list(record.transcript)


async def _generate_responses_object(
    request: NormalizedTurnRequest,
    *,
    response_id: str,
    created_at: int,
    request_log_id: str | None = None,
) -> ResponseObject:
    result = await _generate_turn_result(request, request_log_id=request_log_id)
    output = build_response_output_items(result)
    _store_response_result(
        response_id=response_id,
        created_at=created_at,
        request=request,
        result=result,
        output=output,
    )
    return _build_response_object(
        response_id=response_id,
        created_at=created_at,
        request=request,
        result=result,
        output=output,
    )


async def _responses_stream_response(
    request: NormalizedTurnRequest,
    *,
    client_request: Request | None = None,
    request_log_id: str | None = None,
) -> AsyncGenerator[str, None]:
    response_id = _new_prefixed_id("resp")
    created_at = int(time.time())
    sequence_number = 0
    output_items: list[dict[str, Any]] = []
    message_item: dict[str, Any] | None = None
    message_output_index: int | None = None
    message_closed = False

    def _event(payload: dict[str, Any]) -> str:
        nonlocal sequence_number
        payload["sequence_number"] = sequence_number
        sequence_number += 1
        return _sse(payload)

    def _current_response_payload(
        *,
        status: str,
        result: NormalizedTurnResult | None = None,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if result is None:
            result = NormalizedTurnResult(
                model=_response_model_id(request.model),
                content=None,
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage=NormalizedUsage(0, 0, 0),
            )
        return _build_response_object(
            response_id=response_id,
            created_at=created_at,
            request=request,
            result=result,
            status=status,
            output=output_items,
            error=error,
        ).model_dump()

    yield _event(
        {
            "type": "response.created",
            "response": _current_response_payload(status="in_progress"),
        }
    )
    yield _event(
        {
            "type": "response.in_progress",
            "response": _current_response_payload(status="in_progress"),
        }
    )

    async for event in _stream_events(
        request,
        client_request=client_request,
        request_log_id=request_log_id,
    ):
        if event.kind == "text_delta" and event.text is not None:
            if message_item is None:
                message_item = {
                    "id": _new_prefixed_id("msg"),
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                }
                message_output_index = len(output_items)
                output_items.append(message_item)
                yield _event(
                    {
                        "type": "response.output_item.added",
                        "output_index": message_output_index,
                        "item": message_item,
                    }
                )
                part = {"type": "output_text", "text": "", "annotations": []}
                message_item["content"].append(part)
                yield _event(
                    {
                        "type": "response.content_part.added",
                        "output_index": message_output_index,
                        "item_id": message_item["id"],
                        "content_index": 0,
                        "part": part,
                    }
                )

            message_item["content"][0]["text"] += event.text
            yield _event(
                {
                    "type": "response.output_text.delta",
                    "output_index": message_output_index,
                    "item_id": message_item["id"],
                    "content_index": 0,
                    "delta": event.text,
                    "logprobs": [],
                }
            )
            continue

        if event.kind == "function_call" and event.tool_call is not None:
            function_item = {
                "id": _new_prefixed_id("fc"),
                "type": "function_call",
                "status": "in_progress",
                "call_id": event.tool_call.id,
                "name": event.tool_call.name,
                "arguments": "",
            }
            output_index = len(output_items)
            output_items.append(function_item)
            yield _event(
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": function_item,
                }
            )
            function_item["arguments"] = event.tool_call.arguments
            yield _event(
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": output_index,
                    "item_id": function_item["id"],
                    "delta": event.tool_call.arguments,
                }
            )
            function_item["status"] = "completed"
            yield _event(
                {
                    "type": "response.function_call_arguments.done",
                    "output_index": output_index,
                    "item_id": function_item["id"],
                    "arguments": event.tool_call.arguments,
                    "name": event.tool_call.name,
                }
            )
            yield _event(
                {
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": function_item,
                }
            )
            continue

        if event.kind == "failed" and event.result is not None:
            failed_response = _current_response_payload(
                status="failed",
                result=event.result,
                error={
                    "message": event.error or "Streaming worker failed",
                    "type": "server_error",
                },
            )
            yield _event(
                {
                    "type": "response.failed",
                    "response": failed_response,
                }
            )
            return

        if event.kind == "completed" and event.result is not None:
            if message_item is None and not output_items:
                message_item = {
                    "id": _new_prefixed_id("msg"),
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                }
                message_output_index = len(output_items)
                output_items.append(message_item)
                yield _event(
                    {
                        "type": "response.output_item.added",
                        "output_index": message_output_index,
                        "item": message_item,
                    }
                )
                part = {"type": "output_text", "text": "", "annotations": []}
                message_item["content"].append(part)
                yield _event(
                    {
                        "type": "response.content_part.added",
                        "output_index": message_output_index,
                        "item_id": message_item["id"],
                        "content_index": 0,
                        "part": part,
                    }
                )

            if message_item is not None and not message_closed:
                yield _event(
                    {
                        "type": "response.output_text.done",
                        "output_index": message_output_index,
                        "item_id": message_item["id"],
                        "content_index": 0,
                        "text": message_item["content"][0]["text"],
                        "logprobs": [],
                    }
                )
                yield _event(
                    {
                        "type": "response.content_part.done",
                        "output_index": message_output_index,
                        "item_id": message_item["id"],
                        "content_index": 0,
                        "part": message_item["content"][0],
                    }
                )
                message_item["status"] = "completed"
                yield _event(
                    {
                        "type": "response.output_item.done",
                        "output_index": message_output_index,
                        "item": message_item,
                    }
                )
                message_closed = True

            _store_response_result(
                response_id=response_id,
                created_at=created_at,
                request=request,
                result=event.result,
                output=output_items,
            )
            yield _event(
                {
                    "type": "response.completed",
                    "response": _current_response_payload(
                        status="completed",
                        result=event.result,
                    ),
                }
            )
            return


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

_model: HarmonyModel | None = None
_queue = RequestQueue()
_response_store = ResponseStore()
_stream_active_requests = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _queue.start()
    yield
    await _queue.stop()
    _executor.shutdown(wait=False)


app = FastAPI(title="MLX GPT-OSS Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def http_access_log_middleware(request: Request, call_next):
    if not _HTTP_ACCESS_LOG_ENABLED:
        return await call_next(request)

    request_log_id = _request_log_id(request)
    request_start_ts = time.time()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed_ms = (time.time() - request_start_ts) * 1000.0
        logger.opt(colors=True).info(
            "<n><magenta>{}</magenta>| <bold>rid=<cyan>{}</cyan> method={} path=<white>{}</white> status={} dur_ms=<green>{:.1f}</green></bold></n>",
            _event_label("HTTP"),
            request_log_id,
            request.method,
            request.url.path,
            status_code,
            elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> HealthCheckStatus:
    stats = _queue.stats()
    stats["active_requests"] += _stream_active_requests
    if _model is None:
        return HealthCheckStatus(status="unhealthy", model_id=None, **stats)
    return HealthCheckStatus(status="ok", model_id=_model.model_path, **stats)


@app.get("/v1/models")
async def models_list() -> ModelsResponse:
    if _model is None:
        return ModelsResponse(data=[])
    return ModelsResponse(data=[Model(id=_model.model_path, created=int(time.time()))])


@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, http_request: Request):
    try:
        request_log_id = _request_log_id(http_request)
        logger.opt(colors=True).debug(
            "<yellow>{}</yellow>| rid=<cyan>{}</cyan> stream={} msg={} tools={} maxtokens={}{}",
            _event_label("REQUEST"),
            request_log_id,
            int(bool(payload.stream)),
            len(payload.messages),
            len(payload.tools) if payload.tools else 0,
            payload.max_tokens or _DEFAULT_MAX_TOKENS,
            f" seed={payload.seed}" if payload.seed is not None else "",
        )

        if _model is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {"message": "Model not loaded", "type": "server_error"}
                },
            )

        if payload.stream:
            return StreamingResponse(
                _stream_response(
                    payload,
                    client_request=http_request,
                    request_log_id=request_log_id,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming: run in queue.
        result = await _queue.submit(
            _generate_response,
            payload,
            request_log_id=request_log_id,
        )
        return result
    except Exception:
        logger.exception("Unhandled /v1/chat/completions error")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error",
                    "type": "server_error",
                }
            },
        )


@app.post("/v1/responses")
async def responses_create(payload: ResponsesCreateRequest, http_request: Request):
    request_log_id = _request_log_id(http_request)
    try:
        if _model is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {"message": "Model not loaded", "type": "server_error"}
                },
            )

        previous_transcript = _load_previous_transcript(payload.previous_response_id)
        normalized_request = normalize_responses_request(
            payload,
            previous_transcript=previous_transcript,
        )

        logger.opt(colors=True).debug(
            "<yellow>{}</yellow>| rid=<cyan>{}</cyan> stream={} input_items={} tools={} maxtokens={}{}",
            _event_label("RESPONSE_REQ"),
            request_log_id,
            int(bool(normalized_request.stream)),
            len(normalized_request.input_items),
            len(normalized_request.tools) if normalized_request.tools else 0,
            normalized_request.max_tokens or _DEFAULT_MAX_TOKENS,
            f" seed={normalized_request.seed}" if normalized_request.seed is not None else "",
        )

        if normalized_request.stream:
            return StreamingResponse(
                _responses_stream_response(
                    normalized_request,
                    client_request=http_request,
                    request_log_id=request_log_id,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        response_id = _new_prefixed_id("resp")
        created_at = int(time.time())
        return await _queue.submit(
            _generate_responses_object,
            normalized_request,
            response_id=response_id,
            created_at=created_at,
            request_log_id=request_log_id,
        )
    except InvalidRequestError as exc:
        return _invalid_request_response(exc.message, param=exc.param)
    except Exception:
        logger.exception("Unhandled /v1/responses error")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error",
                    "type": "server_error",
                }
            },
        )


@app.get("/v1/responses/{response_id}")
async def responses_retrieve(response_id: str):
    record = _response_store.get(response_id)
    if record is None:
        return _not_found_response(
            f"Response `{response_id}` was not found.",
            param="response_id",
        )
    return _response_object_from_record(record)


@app.delete("/v1/responses/{response_id}")
async def responses_delete(response_id: str):
    deleted = _response_store.delete(response_id)
    if not deleted:
        return _not_found_response(
            f"Response `{response_id}` was not found.",
            param="response_id",
        )
    return ResponseDeleted(id=response_id)


@app.get("/v1/responses/{response_id}/input_items")
async def responses_input_items(response_id: str):
    record = _response_store.get(response_id)
    if record is None:
        return _not_found_response(
            f"Response `{response_id}` was not found.",
            param="response_id",
        )
    item_ids = [item.get("id") for item in record.input_items if item.get("id")]
    return ResponseInputItemsList(
        data=record.input_items,
        first_id=item_ids[0] if item_ids else None,
        last_id=item_ids[-1] if item_ids else None,
        has_more=False,
    )


# ---------------------------------------------------------------------------
# CLI & entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description="MLX GPT-OSS — OpenAI-compatible Harmony model server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, help="Model path or HuggingFace ID")
    p.add_argument("--host", default="0.0.0.0", help="Bind address")
    p.add_argument("--port", type=int, default=8000, help="Bind port")
    p.add_argument(
        "--context-length",
        type=int,
        default=8196,
        help="Max KV cache context length (total conversation window)",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Minimum log level for console and optional file logging",
    )
    p.add_argument(
        "--log-file",
        default=None,
        help="Path to log file (file logging disabled by default)",
    )
    p.add_argument(
        "--debug-raw-preview-chars",
        type=int,
        default=0,
        help=("When --log-level DEBUG, log up to N chars of rendered prompts"),
    )
    p.add_argument(
        "--http-access-log",
        action="store_true",
        help="Emit one INFO access log line per HTTP request",
    )
    p.add_argument(
        "--responses-store-max-items",
        type=int,
        default=256,
        help="Maximum number of stored /v1/responses records kept in memory",
    )
    p.add_argument(
        "--responses-store-max-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Maximum approximate in-memory size of stored /v1/responses records",
    )
    args = p.parse_args(argv)
    if args.debug_raw_preview_chars < 0:
        p.error("--debug-raw-preview-chars must be >= 0")
    if args.responses_store_max_items < 1:
        p.error("--responses-store-max-items must be >= 1")
    if args.responses_store_max_bytes < 1024:
        p.error("--responses-store-max-bytes must be >= 1024")
    return args


def _print_startup_banner() -> None:
    """Print a compact ASCII startup banner."""
    banner = (
        "\n"
        "\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557\n"
        "\u2551  MLX GPT-OSS Server                  \u2551\n"
        "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d\n"
    )
    sys.stderr.write(banner)


def main(argv: list[str] | None = None) -> None:
    global _DEBUG_RAW_PREVIEW_CHARS
    global _HTTP_ACCESS_LOG_ENABLED
    global _STARTUP_READY_SUMMARY
    global _STARTUP_DEBUG_ARGS
    global _model
    global _response_store

    args = parse_args(argv)
    configure_logging(args.log_level, args.log_file)
    
    _HTTP_ACCESS_LOG_ENABLED = args.http_access_log
    _DEBUG_RAW_PREVIEW_CHARS = args.debug_raw_preview_chars
    _STARTUP_DEBUG_ARGS = (
        "args("
        f"model={args.model}, "
        f"bind={args.host}:{args.port}, "
        f"context_length={args.context_length}, "
        f"log_level={args.log_level}, "
        f"log_file={args.log_file or '-'}, "
        f"debug_raw_preview_chars={args.debug_raw_preview_chars}, "
        f"http_access_log={'true' if args.http_access_log else 'false'}, "
        f"responses_store_max_items={args.responses_store_max_items}, "
        f"responses_store_max_bytes={args.responses_store_max_bytes}"
        ")"
    )
    
    _print_startup_banner()
        
    if _CURRENT_LOG_LEVEL == "DEBUG":
        logger.debug("{}", _STARTUP_DEBUG_ARGS)
            
    logger.opt(colors=True).info("Requesting GPT-OSS model")
    _model = HarmonyModel(args.model, context_length=args.context_length)
    _response_store = ResponseStore(
        max_items=args.responses_store_max_items,
        max_bytes=args.responses_store_max_bytes,
    )
    logger.opt(colors=True).info("Model loaded and ready")
    logger.opt(colors=True).info("Listening on <cyan>{}:{}</cyan>", args.host, args.port)
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
