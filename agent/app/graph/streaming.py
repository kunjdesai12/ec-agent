"""Streaming turn executor.

We can't naively stream the LangGraph because intermediate tool-call turns
need to fully resolve before we know whether to loop. Strategy:

  1. Run the graph normally up to the point where the LLM decides "no more
     tools" — i.e. produces a final text response.
  2. Throw that final text away.
  3. Re-issue the same final LLM call with stream=True, using the message
     state at that point (which includes all the tool results).

This is slightly more compute (one extra non-stream completion), but it
keeps the graph logic clean and gives true token-by-token streaming for
the user-visible response. For sub-second-latency requirements you'd
restructure to stream directly from the graph; we can optimize later.
"""
from __future__ import annotations

from typing import Any, AsyncIterator

from agent.app.graph.orchestrator import build_initial_messages, get_graph
from agent.app.llm import stream_text
from agent.app.logging_setup import get_logger
from agent.app.rag import format_chunks_for_prompt, get_retriever
from agent.app.tools import TOOL_SCHEMAS

log = get_logger(__name__)


async def stream_turn(
    session_id: str,
    user_message: str,
    history: list[dict[str, Any]],
    order_params: Optional[dict[str, Any]] = None,
    jwt_token: str = "",
) -> AsyncIterator[dict[str, Any]]:
    """Yield SSE-friendly events for a turn.

    Event shapes:
      {"type": "status", "stage": "retrieving" | "thinking" | "calling_tool" | "generating"}
      {"type": "tool",   "name": str, "args": dict}
      {"type": "token",  "text": str}
      {"type": "done",   "messages": [...persistable msgs...]}
      {"type": "error",  "message": str}
    """
    try:
        # Stage 1: run the graph to completion (non-streamed)
        yield {"type": "status", "stage": "retrieving"}
        graph = get_graph()
        initial_messages = build_initial_messages(history, user_message)
        final_state = await graph.ainvoke(
            {
                "session_id": session_id,
                "user_message": user_message,
                "messages": initial_messages,
                "tool_iterations": 0,
            }
        )

        # Surface which tools were used as informational events
        for m in final_state["messages"]:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    yield {
                        "type": "tool",
                        "name": tc["function"]["name"],
                        "args": tc["function"]["arguments"],
                    }

        # Stage 2: re-stream the final assistant turn for true token streaming.
        # We strip the trailing assistant message (which has the final text)
        # and ask the model to regenerate with stream=True. Tools are still
        # provided so the model behaves identically.
        all_msgs = final_state["messages"]

        # Drop trailing assistant text-only message so the model produces it fresh
        stream_input = list(all_msgs)
        if (
            stream_input
            and stream_input[-1].get("role") == "assistant"
            and not stream_input[-1].get("tool_calls")
        ):
            stream_input = stream_input[:-1]

        yield {"type": "status", "stage": "generating"}
        streamed_text_parts: list[str] = []
        async for delta in stream_text(stream_input, tools=TOOL_SCHEMAS):
            streamed_text_parts.append(delta)
            yield {"type": "token", "text": delta}

        # Build the persistence payload: original history + this turn's new messages.
        # We use the streamed text as the canonical final assistant message
        # (might differ slightly from the non-streamed first pass due to sampling).
        new_messages: list[dict[str, Any]] = []
        for m in final_state["messages"]:
            if m.get("role") == "system":
                continue
            if (
                m.get("role") == "assistant"
                and not m.get("tool_calls")
                and m is final_state["messages"][-1]
            ):
                # Replace the non-streamed final with the streamed one
                continue
            new_messages.append(m)
        new_messages.append({"role": "assistant", "content": "".join(streamed_text_parts)})

        yield {"type": "done", "messages": new_messages}

    except Exception as e:  # noqa: BLE001
        log.exception("stream_turn_failed")
        yield {"type": "error", "message": str(e)}
