"""
Harmony protocol for GPT-OSS models.
Prompt rendering and response parsing using the official openai_harmony library.
Various fixes to avoid empty responses and proper tool calling.
"""

from __future__ import annotations

import datetime
import json
import re
from enum import Enum
from typing import Any

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    SystemContent,
    TextContent,
    ToolDescription,
    load_harmony_encoding,
)

# ---------------------------------------------------------------------------
# Encoding singleton
# ---------------------------------------------------------------------------

_encoding = None


def get_encoding():
    """Return cached Harmony encoding for GPT-OSS."""
    global _encoding
    if _encoding is None:
        _encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _encoding


def get_stop_tokens() -> list[int]:
    """Get the stop tokens for assistant actions from the encoding.

    Uses the official openai_harmony API rather than hardcoding token IDs.
    """
    return list(get_encoding().stop_tokens_for_assistant_actions())


# ---------------------------------------------------------------------------
# Prompt rendering  (OpenAI messages -> Harmony token string)
# ---------------------------------------------------------------------------

REASONING_EFFORTS = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}


def _convert_tools(tools: list[dict[str, Any]]) -> list[ToolDescription]:
    """Convert OpenAI-format tool definitions to Harmony ToolDescription list."""
    harmony_tools = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        harmony_tools.append(
            ToolDescription.new(
                name=func.get("name", ""),
                description=func.get("description", ""),
                parameters=func.get("parameters", {}),
            )
        )
    return harmony_tools


def _coerce_message_content_to_text(content: Any) -> str:
    """
    Normalize OpenAI/OpenWebUI message ``content`` payloads into plain text.

    Supports:
    - plain strings
    - multimodal content arrays (e.g. [{"type":"text"...}, {"type":"image_url"...}])
    - dict payloads with text-like fields

    Non-text blocks are ignored unless no text exists, in which case we keep a
    short placeholder so the turn is not silently dropped.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool)):
        return str(content)

    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        ctype = content.get("type")
        if ctype in {"image_url", "image", "input_image"}:
            return "[non-text content omitted]"
        return json.dumps(content, ensure_ascii=False)

    if not isinstance(content, list):
        return str(content)

    text_parts: list[str] = []
    saw_non_text = False
    for part in content:
        if isinstance(part, str):
            if part:
                text_parts.append(part)
            continue

        if not isinstance(part, dict):
            saw_non_text = True
            continue

        ptype = part.get("type")
        if ptype in {"text", "input_text"}:
            text = part.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
            continue

        if ptype in {"image_url", "image", "input_image"}:
            saw_non_text = True
            continue

        # Best-effort extraction for provider-specific fields.
        text = part.get("text")
        if isinstance(text, str) and text:
            text_parts.append(text)
        else:
            saw_non_text = True

    if text_parts:
        return "\n".join(text_parts)
    if saw_non_text:
        return "[non-text content omitted]"
    return ""


def render_harmony_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    reasoning_effort: str = "medium",
) -> str:
    """Render OpenAI-style messages into a Harmony-encoded prompt string.

    Parameters
    ----------
    messages : list
        OpenAI-format messages (role/content/tool_calls/etc.).
    tools : list, optional
        OpenAI-format tool definitions.
    reasoning_effort : str
        ``"low"``, ``"medium"``, or ``"high"``.

    Returns
    -------
    str
        Properly encoded Harmony prompt ready for tokenisation.
    """
    encoding = get_encoding()
    harmony_messages: list[Message] = []

    # Get system instructions from user.
    system_instructions: list[str] = []
    for msg in messages:
        if msg.get("role") == "system":
            content = _coerce_message_content_to_text(msg.get("content", ""))
            if content:
                system_instructions.append(content)

    # System message (Harmony metadata) + add current date.
    effort = REASONING_EFFORTS.get(reasoning_effort, ReasoningEffort.MEDIUM)
    system_content = (
        SystemContent.new()
        .with_reasoning_effort(effort)
        .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
    )
    harmony_messages.append(Message.from_role_and_content(Role.SYSTEM, system_content))

    # Developer message (system instructions + tool definitions).
    combined_instructions = "\n\n".join(system_instructions)
    if tools or combined_instructions:
        dev_content = DeveloperContent.new()
        if combined_instructions:
            dev_content = dev_content.with_instructions(combined_instructions)
        if tools:
            harmony_tools = _convert_tools(tools)
            if harmony_tools:
                dev_content = dev_content.with_function_tools(harmony_tools)
        harmony_messages.append(
            Message.from_role_and_content(Role.DEVELOPER, dev_content)
        )

    # Conversation messages.
    for msg in messages:
        role = msg.get("role", "user")
        content = _coerce_message_content_to_text(msg.get("content", ""))

        if role == "system":
            # Already merged into the developer message above.
            continue

        elif role == "developer":
            if content:
                dev = DeveloperContent.new().with_instructions(content)
                harmony_messages.append(
                    Message.from_role_and_content(Role.DEVELOPER, dev)
                )

        elif role == "user":
            if content:
                harmony_messages.append(
                    Message.from_role_and_content(Role.USER, content)
                )

        elif role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    func_name = func.get("name", "")
                    func_args = func.get("arguments", "{}")
                    if not isinstance(func_args, str):
                        func_args = json.dumps(func_args)
                    tool_msg = (
                        Message(
                            author=Author.new(Role.ASSISTANT, "assistant"),
                            content=[TextContent(text=func_args)],
                        )
                        .with_recipient(f"functions.{func_name}")
                        .with_channel("commentary")
                    )
                    harmony_messages.append(tool_msg)
            elif content:
                harmony_messages.append(
                    Message.from_role_and_content(Role.ASSISTANT, content)
                )

        elif role == "tool":
            tool_name = msg.get("name", "")
            tool_call_id = msg.get("tool_call_id", "")

            # Resolve tool name from tool_call_id if needed.
            if not tool_name and tool_call_id:
                for prev in messages:
                    for tc in prev.get("tool_calls", []):
                        if tc.get("id") == tool_call_id:
                            tool_name = tc.get("function", {}).get("name", "")
                            break

            if not tool_name:
                tool_name = "unknown_function"

            tool_response = (
                Message(
                    author=Author.new(Role.TOOL, f"functions.{tool_name}"),
                    content=[TextContent(text=content or "")],
                )
                .with_recipient("assistant")
                .with_channel("commentary")
            )
            harmony_messages.append(tool_response)

    # Return formatted tokens.
    conversation = Conversation.from_messages(harmony_messages)
    tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
    return encoding.decode(tokens)


# ---------------------------------------------------------------------------
# Response parsing  (model output -> content / tool_calls / reasoning)
# ---------------------------------------------------------------------------


class _Channel(Enum):
    ANALYSIS = "analysis"
    COMMENTARY = "commentary"
    FINAL = "final"


class _ParserState(Enum):
    NORMAL = "normal"
    FOUND_ARGUMENTS = "found_arguments"
    END_STREAM = "end_stream"


class HarmonyParser:
    """
    Parse Harmony-encoded model output -> structured results.
    Handles streaming and non-streaming modes.
    """

    def __init__(self) -> None:
        self.encoding = get_encoding()
        self.parser = StreamableParser(self.encoding, role=Role.ASSISTANT)
        self.end_tool_chunk = "<|call|>"
        self.state = _ParserState.NORMAL
        self.arguments_buffer: list[str] = []
        self.function_name_buffer = ""

    @staticmethod
    def _normalize_tool_name(name_or_recipient: str | None) -> str | None:
        if not name_or_recipient:
            return None

        candidate = name_or_recipient.strip()
        if "functions." in candidate:
            candidate = candidate.split("functions.", 1)[1]

        marker_idx = candidate.find("<|")
        if marker_idx != -1:
            candidate = candidate[:marker_idx]

        candidate = candidate.strip().strip(" \t\r\n.,:;\"'")
        match = re.match(r"[A-Za-z_][A-Za-z0-9_.-]*", candidate)
        if not match:
            return None
        return match.group(0)

    def _parse_with_encoding(self, text: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "content": None,
            "tool_calls": [],
            "reasoning_content": None,
        }

        tokens = self.encoding.encode(text, allowed_special="all")
        parsed_messages = self.encoding.parse_messages_from_completion_tokens(
            tokens, role=Role.ASSISTANT
        )

        for message in parsed_messages:
            recipient = getattr(message, "recipient", "") or ""
            tool_name = self._normalize_tool_name(recipient)
            is_tool_call = message.channel == _Channel.COMMENTARY.value or (
                tool_name is not None
            )

            if is_tool_call and tool_name:
                result["tool_calls"].append(
                    {
                        "name": tool_name,
                        "arguments": message.content[0].text,
                    }
                )
            elif message.channel == _Channel.ANALYSIS.value:
                result["reasoning_content"] = message.content[0].text
            elif message.channel == _Channel.FINAL.value:
                result["content"] = message.content[0].text

        return result

    def _candidate_completion_texts(self, text: str) -> list[str]:
        marker = "<|start|>assistant"
        candidates = [text]

        # Some outputs include extra wrapper tokens.
        # Try a few safe text slices first before using regex fallback.
        if marker in text:
            candidates.append(text.replace(marker, ""))

        channel_idx = text.find("<|channel|>")
        if channel_idx >= 0:
            candidates.append(text[channel_idx:])

        assistant_channel_idx = text.find(f"{marker}<|channel|>")
        if assistant_channel_idx >= 0:
            candidates.append(text[assistant_channel_idx + len(marker) :])

        unique: list[str] = []
        seen = set()
        for candidate in candidates:
            if not candidate:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            unique.append(candidate)
        return unique

    @staticmethod
    def _extract_balanced_block(text: str, start_idx: int) -> str | None:
        if start_idx >= len(text):
            return None
        opening = text[start_idx]
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escape = False

        for idx in range(start_idx, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    return text[start_idx : idx + 1]
        return None

    @staticmethod
    def _slice_to_next_marker(
        text: str,
        start_idx: int,
        *,
        stop_at_newline: bool = True,
    ) -> str:
        end = len(text)
        markers = [
            "<|start|>",
            "<|channel|>",
            "<|message|>",
            "<|constrain|>",
            "<|end|>",
            "<|return|>",
            "<|call|>",
        ]
        if stop_at_newline:
            markers.append("\n")

        for marker in markers:
            idx = text.find(marker, start_idx)
            if idx != -1 and idx < end:
                end = idx
        return text[start_idx:end].strip()

    def _extract_tool_payload(self, text: str, search_start: int) -> str | None:
        # Use <|message|> when it exists. If not, start at the first { or [.
        msg_idx = text.find("<|message|>", search_start)
        if msg_idx != -1:
            payload_start = msg_idx + len("<|message|>")
        else:
            brace_positions = [
                idx
                for idx in (text.find("{", search_start), text.find("[", search_start))
                if idx != -1
            ]
            if not brace_positions:
                return None
            payload_start = min(brace_positions)

        while payload_start < len(text) and text[payload_start] in " \t\r\n:=":
            payload_start += 1
        if payload_start >= len(text):
            return None

        start_char = text[payload_start]
        if start_char in "{[":
            block = self._extract_balanced_block(text, payload_start)
            if block:
                return block.strip()
            # If JSON is cut off, take text until the next Harmony marker.
            return self._slice_to_next_marker(text, payload_start) or None

        return self._slice_to_next_marker(text, payload_start) or None

    @staticmethod
    def _current_channel_at(text: str, pos: int) -> str | None:
        marker = "<|channel|>"
        idx = text.rfind(marker, 0, pos)
        if idx == -1:
            return None
        start = idx + len(marker)
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text):
            return None
        end = start
        while end < len(text):
            if text[end].isspace() or text.startswith("<|", end):
                break
            end += 1
        if end <= start:
            return None
        return text[start:end]

    def _extract_tool_calls_fallback(self, text: str) -> list[dict[str, str]]:
        calls: list[dict[str, str]] = []
        recipient_pattern = re.compile(r"to=functions\.([A-Za-z0-9_.-]+)")

        for match in recipient_pattern.finditer(text):
            name = self._normalize_tool_name(match.group(1).strip())
            payload = self._extract_tool_payload(text, match.end())
            if not name:
                continue

            if not payload:
                # Recovery path for malformed tool call headers that omit
                # `<|message|>{...}` but still indicate a real tool invocation.
                channel = self._current_channel_at(text, match.start())
                header_end = min(len(text), match.end() + 160)
                has_tool_header = (
                    "<|constrain|>" in text[match.end() : header_end]
                    or "<|call|>" in text[match.end() : header_end]
                )
                if (
                    has_tool_header
                    and channel in {_Channel.ANALYSIS.value, _Channel.COMMENTARY.value}
                ):
                    payload = "{}"
                else:
                    continue

            calls.append({"name": name, "arguments": payload})

        deduped: list[dict[str, str]] = []
        seen = set()
        for call in calls:
            key = (call["name"], call["arguments"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(call)
        return deduped

    @staticmethod
    def _extract_channel_segments(text: str, channel: str) -> list[str]:
        pattern = re.compile(
            rf"<\|channel\|>\s*{re.escape(channel)}\b.*?<\|message\|>",
            re.DOTALL,
        )
        segments: list[str] = []
        for match in pattern.finditer(text):
            start = match.end()
            segment = HarmonyParser._slice_to_next_marker(
                text,
                start,
                stop_at_newline=False,
            )
            if segment:
                segments.append(segment)
        return segments

    @staticmethod
    def _strip_harmony_markup(text: str) -> str:
        cleaned = re.sub(r"<\|[^|]+\|>", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _parse_fallback(self, text: str) -> dict[str, Any]:
        tool_calls = self._extract_tool_calls_fallback(text)
        reasoning_segments = self._extract_channel_segments(
            text, _Channel.ANALYSIS.value
        )
        final_segments = self._extract_channel_segments(text, _Channel.FINAL.value)
        has_harmony_markup = any(
            marker in text
            for marker in (
                "<|start|>",
                "<|channel|>",
                "<|message|>",
                "<|end|>",
            )
        )

        reasoning_content = "\n".join(reasoning_segments).strip() or None
        content = "\n".join(final_segments).strip() or None

        if not content and not tool_calls:
            if has_harmony_markup:
                # Avoid showing internal channels (analysis/commentary) as regular message to the user.
                content = None
            else:
                content = self._strip_harmony_markup(text) or None
        elif tool_calls and not final_segments:
            # Avoid returning internal planning text when we're issuing tool calls.
            content = None

        return {
            "content": content,
            "tool_calls": tool_calls,
            "reasoning_content": reasoning_content,
        }

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    def parse(self, text: str) -> dict[str, Any]:
        """Parse complete model output text.

        Returns
        -------
        dict
            ``{"content": ..., "tool_calls": [...], "reasoning_content": ...}``
        """
        # `<|call|>` marks end of the assistant action. Cut anything after it.
        if self.end_tool_chunk in text:
            idx = text.find(self.end_tool_chunk)
            text = text[: idx + len(self.end_tool_chunk)]

        for candidate in self._candidate_completion_texts(text):
            try:
                return self._parse_with_encoding(candidate)
            except Exception:
                continue

        return self._parse_fallback(text)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _build_result(
        self,
        reasoning: list[str],
        tool_calls: list[dict[str, str]] | None,
        contents: list[str],
    ) -> dict[str, Any]:
        return {
            "reasoning_content": "".join(reasoning) or None,
            "tool_calls": tool_calls,
            "content": "".join(contents) or None,
        }

    def parse_streaming(
        self,
        chunk: str,
        *,
        parse_tools: bool = True,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Parse a single streaming chunk.

        Returns
        -------
        tuple
            ``(parsed_result_or_None, is_complete)``
        """
        if self.state == _ParserState.END_STREAM:
            return None, True

        # Streaming states:
        # NORMAL -> FOUND_ARGUMENTS (collect tool args) -> END_STREAM.
        reasoning: list[str] = []
        contents: list[str] = []
        end_stream = False

        # Detect and truncate at the call marker.
        if self.end_tool_chunk in chunk:
            idx = chunk.find(self.end_tool_chunk)
            chunk = chunk[: idx + len(self.end_tool_chunk)]
            end_stream = True

        # Tokenise and feed through the streaming parser.
        chunk_tokens = self.encoding.encode(chunk, allowed_special="all")
        for token in chunk_tokens:
            stream_text = self.parser.process(token)
            delta = stream_text.last_content_delta
            if not delta:
                continue

            if parse_tools and self.state == _ParserState.FOUND_ARGUMENTS:
                # After a tool call starts, later deltas are tool arguments.
                self.arguments_buffer.append(delta)
                continue

            channel = stream_text.current_channel
            if parse_tools:
                recipient = stream_text.current_recipient or ""
                tool_name = self._normalize_tool_name(recipient)
                is_tool_call = channel == _Channel.COMMENTARY.value or (
                    tool_name is not None
                )

                if is_tool_call and tool_name:
                    self.state = _ParserState.FOUND_ARGUMENTS
                    self.arguments_buffer.append(delta)
                    self.function_name_buffer = tool_name
                elif channel == _Channel.ANALYSIS.value:
                    reasoning.append(delta)
                elif channel == _Channel.FINAL.value:
                    contents.append(delta)
            else:
                if channel == _Channel.ANALYSIS.value:
                    reasoning.append(delta)
                elif channel == _Channel.FINAL.value:
                    contents.append(delta)

        if end_stream:
            tool_calls: list[dict[str, str]] = []
            if parse_tools and self.function_name_buffer:
                tool_calls.append(
                    {
                        "name": self.function_name_buffer,
                        "arguments": "".join(self.arguments_buffer),
                    }
                )
            self.arguments_buffer = []
            self.function_name_buffer = ""
            self.state = _ParserState.END_STREAM
            return (
                self._build_result(
                    reasoning,
                    (tool_calls or None) if parse_tools else None,
                    contents,
                ),
                True,
            )

        return self._build_result(reasoning, None, contents), False

    def handle_stream_end(
        self,
        *,
        parse_tools: bool = True,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Flush any remaining buffered tool-call data at end of stream."""
        if parse_tools and self.state == _ParserState.FOUND_ARGUMENTS:
            # Some streams end without `<|call|>`. Add it so normal finish logic runs.
            return self.parse_streaming(self.end_tool_chunk, parse_tools=True)
        return None, False
