"""vLLM client wrapper using the OpenAI SDK.

vLLM exposes an OpenAI-compatible /v1/chat/completions endpoint. With
--enable-auto-tool-choice --tool-call-parser hermes, the server extracts
Hermes <tool_call> blocks from Qwen's raw output and returns them as
structured `tool_calls` in the response — exactly like OpenAI.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Optional

from openai import AsyncOpenAI

from agent.app.config import get_settings
from agent.app.logging_setup import get_logger

log = get_logger(__name__)

_client: Optional[AsyncOpenAI] = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        s = get_settings()
        _client = AsyncOpenAI(
            base_url=s.vllm_base_url,
            api_key=s.vllm_api_key,
            timeout=s.vllm_request_timeout_s,
        )
    return _client


async def chat_complete(
    messages: list[dict[str, Any]],
    *,
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: str = "auto",
    stream: bool = False,
) -> Any:
    """Non-streaming or streaming chat completion.

    Returns:
      - If stream=False: a ChatCompletion object
      - If stream=True:  an async iterator of ChatCompletionChunk
    """
    s = get_settings()
    client = get_client()

    kwargs: dict[str, Any] = {
        "model": s.vllm_model,
        "messages": messages,
        "max_tokens": s.max_new_tokens,
        "temperature": s.temperature,
        "top_p": s.top_p,
        "stream": stream,
    }
    log.info("llm_messages", extra={
        "messages": messages,
    })
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

    return await client.chat.completions.create(**kwargs)


async def stream_text(
    messages: list[dict[str, Any]],
    *,
    tools: Optional[list[dict[str, Any]]] = None,
) -> AsyncIterator[str]:
    """Stream just the text deltas. Used for the final assistant turn after
    all tool calls have resolved (we don't stream intermediate tool-call
    decision turns; those need to be fully resolved before we know what to do).
    """
    stream = await chat_complete(messages, tools=tools, stream=True)
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content
