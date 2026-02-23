# MLX GPT-OSS Server

Minimal OpenAI-compatible server for GPT-OSS/Harmony models on Apple Silicon.  
Built with `mlx-lm` (inference), `openai-harmony` (prompt formatting), and FastAPI (HTTP API).

## Feature List

- OpenAI-style `/v1/chat/completions` endpoint
- Streaming (`SSE`) and non-streaming responses
- Harmony `reasoning_effort` support (`low`, `medium`, `high`)
- OpenAI tool-calling response format
- Robust Harmony tool-calling parser and stream recovery paths
- Usage token counts in responses
- `/health` queue stats and `/v1/models` compatibility endpoint
- Single-model runtime with FIFO request queueing

## Requirements

- macOS on Apple Silicon
- Python `>=3.11`

## Quick Start

```bash
pip install mlx-gpt-oss
mlx-gpt-oss --model mlx-community/gpt-oss-20b-MXFP4-Q8
```

Default bind: `http://0.0.0.0:8000`

## Install From Source

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
mlx-gpt-oss --model mlx-community/gpt-oss-20b-MXFP4-Q8
```

## API Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | `GET` | Server health + active/queued request counts |
| `/v1/models` | `GET` | Loaded model metadata |
| `/v1/chat/completions` | `POST` | OpenAI-compatible chat completion |

## Chat Completions Notes

- `model` is required for compatibility, but the server always uses the single model loaded at startup.
- Supports OpenAI-style `messages`, `stream`, `tools`, `tool_choice`, `stop`, and common sampling params.
- `top_k` is accepted but generation remains pinned to `top_k=0` for GPT-OSS behavior.
- `reasoning_effort` can be set directly, or via `chat_template_kwargs.reasoning_effort`.
- Streaming returns `chat.completion.chunk` events and ends with `[DONE]`.

## Tool Calling Reliability

- Uses official Harmony assistant-action stop tokens from `openai-harmony` (no hardcoded token IDs).
- Handles streaming edge cases: unfinished tool-call endings, buffered fallback dedupe, and repeated identical tool calls.
- Addresses a class of tool-calling failures seen in other MLX servers.

## CLI Options

| Flag | Default | Description |
| --- | --- | --- |
| `--model` | required | Model path or Hugging Face ID |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Bind port |
| `--context-length` | `8196` | Max KV cache context length |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--log-file` | disabled | Optional rotating file log output |
| `--debug-raw-preview-chars` | `0` | In `DEBUG`, preview N chars of prompts/output |
| `--http-access-log` | `False` | Emit one access log line per HTTP request |

## Security

- No built-in auth or API key checks, this is your responsibility.
- Default host is `0.0.0.0` for local/LAN self-hosting.
- CORS is permissive (`*`, credentials disabled).
- Use `--host 127.0.0.1` for local-only access.
