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
        self._queue: asyncio.Queue[
            tuple[asyncio.Future[Any], Any, tuple[Any, ...], dict[str, Any]]
        ] = asyncio.Queue(maxsize=32)
        self._active = 0
        self._worker: asyncio.Task | None = None

    async def start(self) -> None:
        self._worker = asyncio.create_task(self._loop())
        logger.opt(colors=True).info( "Server is <green>ready</green>" )

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass

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
    request: ChatCompletionRequest,
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
    request: ChatCompletionRequest,
    *,
    request_log_id: str | None = None,
) -> list[int]:
    """Build input_ids from the request messages."""
    reasoning_effort = request.reasoning_effort
    if (
        request.reasoning_effort is None
        and request.chat_template_kwargs
        and isinstance(request.chat_template_kwargs.get("reasoning_effort"), str)
    ):
        reasoning_effort = request.chat_template_kwargs["reasoning_effort"]
    reasoning_effort = _normalize_reasoning_effort(reasoning_effort)

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


async def _generate_response(
    request: ChatCompletionRequest,
    *,
    request_log_id: str | None = None,
) -> ChatCompletionResponse:
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

    # Build response.
    response = _format_response(
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

    return response


def _format_response(
    parsed: dict[str, Any],
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    finish_reason: str = "stop",
) -> ChatCompletionResponse:
    """Format parsed result into an OpenAI chat completion response."""
    tool_calls = parsed.get("tool_calls") or []

    tc_objects = []
    for tc in tool_calls:
        normalized_tool_call = _normalize_tool_call(tc)
        if normalized_tool_call is None:
            continue
        sanitized_name, args = normalized_tool_call
        tc_objects.append(
            ChatCompletionMessageToolCall(
                id=_new_prefixed_id("call"),
                function=FunctionCall(
                    name=sanitized_name,
                    arguments=args,
                ),
                index=len(tc_objects),
            )
        )

    resolved_finish_reason = "tool_calls" if tc_objects else finish_reason

    return ChatCompletionResponse(
        id=_new_prefixed_id("chatcmpl"),
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                message=Message(
                    role="assistant",
                    content=parsed.get("content"),
                    reasoning_content=parsed.get("reasoning_content"),
                    tool_calls=tc_objects or None,
                ),
                finish_reason=resolved_finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
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
    context: _StreamRequestContext,
    *,
    delta: Delta,
    finish_reason: str | None = None,
    usage: UsageInfo | None = None,
) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id=context.chat_id,
        created=context.created,
        model=context.model,
        choices=[StreamingChoice(delta=delta, finish_reason=finish_reason)],
        usage=usage,
    )


def _build_stream_tool_call_chunks(
    tool_calls: list[dict[str, Any]],
    *,
    context: _StreamRequestContext,
    tool_call_index: int,
) -> tuple[list[ChatCompletionChunk], int, list[tuple[str, str]]]:
    chunks: list[ChatCompletionChunk] = []
    emitted_tool_calls: list[tuple[str, str]] = []
    for tool_call in tool_calls:
        normalized_tool_call = _normalize_tool_call(
            tool_call,
            warning_label="stream tool call",
        )
        if normalized_tool_call is None:
            continue

        tool_name, arguments = normalized_tool_call
        emitted_tool_calls.append((tool_name, arguments))
        tool_call_index += 1
        chunks.append(
            _stream_chunk(
                context,
                delta=Delta(
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=tool_call_index,
                            id=_new_prefixed_id("call"),
                            function=ChoiceDeltaFunctionCall(
                                name=tool_name,
                                arguments=arguments,
                            ),
                        )
                    ],
                ),
            )
        )

    return chunks, tool_call_index, emitted_tool_calls


@dataclass
class _StreamState:
    """Mutable fields that evolve while a stream is running."""

    finish_reason: str = "stop"
    completion_tokens: int = 0
    tool_call_index: int = -1
    stop_tail: str = ""
    raw_output_parts: list[str] = field(default_factory=list)
    emitted_content_parts: list[str] = field(default_factory=list)
    emitted_reasoning_parts: list[str] = field(default_factory=list)
    emitted_tool_calls: list[tuple[str, str]] = field(default_factory=list)
    raw_preview_parts: list[str] = field(default_factory=list)
    raw_preview_len: int = 0
    raw_text_fallback: bool = False
    disconnected: bool = False
    stream_cancel_reason: str | None = None


def _build_stream_request_context(
    request: ChatCompletionRequest,
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
    """SSE streaming generator matching the OpenAI protocol."""
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

    # First chunk: role-only delta (OpenAI convention).
    yield _sse(_stream_chunk(context, delta=Delta(role="assistant")))

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
                    yield _sse(
                        _stream_chunk(context, delta=Delta(content=state.stop_tail))
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
                        pending_tool_calls = _drop_emitted_tool_call_prefix(
                            flushed["tool_calls"],
                            state.emitted_tool_calls,
                        )
                        (
                            tool_call_chunks,
                            state.tool_call_index,
                            emitted_tool_calls,
                        ) = (
                            _build_stream_tool_call_chunks(
                                pending_tool_calls,
                                context=context,
                                tool_call_index=state.tool_call_index,
                            )
                        )
                        state.emitted_tool_calls.extend(emitted_tool_calls)
                        if tool_call_chunks:
                            state.finish_reason = "tool_calls"
                        for chunk_item in tool_call_chunks:
                            yield _sse(chunk_item)

                if state.raw_text_fallback and state.raw_output_parts:
                    parsed = _parse_streaming_fallback_output(
                        "".join(state.raw_output_parts),
                        parse_tools=parse_tools,
                    )
                    reasoning_content = parsed.get("reasoning_content")
                    if reasoning_content:
                        reasoning_content = _only_unseen_suffix(
                            reasoning_content,
                            "".join(state.emitted_reasoning_parts),
                        )
                    if reasoning_content:
                        state.emitted_reasoning_parts.append(reasoning_content)
                        yield _sse(
                            _stream_chunk(
                                context,
                                delta=Delta(reasoning_content=reasoning_content),
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
                        yield _sse(
                            _stream_chunk(context, delta=Delta(content=parsed_content))
                        )

                    if parse_tools and parsed.get("tool_calls"):
                        pending_tool_calls = _drop_emitted_tool_call_prefix(
                            parsed["tool_calls"],
                            state.emitted_tool_calls,
                        )
                        (
                            tool_call_chunks,
                            state.tool_call_index,
                            emitted_tool_calls,
                        ) = (
                            _build_stream_tool_call_chunks(
                                pending_tool_calls,
                                context=context,
                                tool_call_index=state.tool_call_index,
                            )
                        )
                        state.emitted_tool_calls.extend(emitted_tool_calls)
                        if tool_call_chunks:
                            state.finish_reason = "tool_calls"
                        for chunk_item in tool_call_chunks:
                            yield _sse(chunk_item)
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
                yield _sse(
                    _stream_chunk(
                        context,
                        delta=Delta(reasoning_content=reasoning_content),
                    )
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
                            yield _sse(
                                _stream_chunk(context, delta=Delta(content=emit_text))
                            )
                        break
                    content_delta = emit_text
                if content_delta:
                    state.emitted_content_parts.append(content_delta)
                    yield _sse(
                        _stream_chunk(context, delta=Delta(content=content_delta))
                    )

            # Emit tool call deltas.
            if parse_tools and parsed.get("tool_calls"):
                tool_call_chunks, state.tool_call_index, emitted_tool_calls = (
                    _build_stream_tool_call_chunks(
                        parsed["tool_calls"],
                        context=context,
                        tool_call_index=state.tool_call_index,
                    )
                )
                state.emitted_tool_calls.extend(emitted_tool_calls)
                if tool_call_chunks:
                    state.finish_reason = "tool_calls"
                for chunk_item in tool_call_chunks:
                    yield _sse(chunk_item)

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

    # Final chunk with finish_reason + usage.
    usage = UsageInfo(
        prompt_tokens=context.prompt_len,
        completion_tokens=state.completion_tokens,
        total_tokens=context.prompt_len + state.completion_tokens,
    )
    final_chunk = _stream_chunk(
        context,
        delta=Delta(),
        finish_reason=state.finish_reason,
        usage=usage,
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
    yield _sse(final_chunk)
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

_model: HarmonyModel | None = None
_queue = RequestQueue()
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
    args = p.parse_args(argv)
    if args.debug_raw_preview_chars < 0:
        p.error("--debug-raw-preview-chars must be >= 0")
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
        f"http_access_log={'true' if args.http_access_log else 'false'}"
        ")"
    )
    
    _print_startup_banner()
        
    if _CURRENT_LOG_LEVEL == "DEBUG":
        logger.debug("{}", _STARTUP_DEBUG_ARGS)
            
    logger.opt(colors=True).info("Requesting GPT-OSS model")
    _model = HarmonyModel(args.model, context_length=args.context_length)
    logger.opt(colors=True).info("Model loaded and ready")
    logger.opt(colors=True).info("Listening on <cyan>{}:{}</cyan>", args.host, args.port)
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
