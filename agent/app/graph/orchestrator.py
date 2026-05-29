"""LangGraph orchestration for Aki.

Flow:
                ┌──────────────┐
   start ───▶   │   retrieve   │  (bge-m3 over menu corpus)
                └──────┬───────┘
                       │  (RAG context injected as a system note)
                       ▼
                ┌──────────────┐
                │     llm      │◀────────────┐
                └──────┬───────┘             │
                       │                     │
              tool_calls?                    │
                 ├─ yes ─▶ ┌──────────────┐ │
                 │         │    tools     │ │
                 │         └──────┬───────┘ │
                 │                └─────────┘
                 └─ no ──▶ END (assistant message ready)

The `llm` node may decide to call tools multiple times in sequence
(e.g. get_user_addresses → place_order). The graph loops llm ⇄ tools
up to max_tool_iterations before giving up and asking the user.

Streaming note: streaming happens in `app/main.py` for the *final* turn
only — after the model decides it's done calling tools. Intermediate
tool-call decision turns are not streamed because we need the full
response to decide whether to loop.
"""
from __future__ import annotations

import json
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import StateGraph, END

from agent.app.config import get_settings
from agent.app.graph.prompts import AKI_SYSTEM_PROMPT
from agent.app.llm import chat_complete
from agent.app.logging_setup import get_logger
from agent.app.rag import format_chunks_for_prompt, get_retriever
from agent.app.tools import TOOL_SCHEMAS, execute_tool

log = get_logger(__name__)


def _merge_messages(left: list[dict], right: list[dict]) -> list[dict]:
    """Reducer: append new messages to existing history."""
    return left + right


class GraphState(TypedDict, total=False):
    session_id: str
    user_message: str
    # Full message list passed to the LLM — system + history + new user msg + tool turns
    messages: Annotated[list[dict[str, Any]], _merge_messages]
    rag_context: str
    tool_iterations: int
    final_text: str   # populated when we exit the loop


# ────────────────────────────────────────────────────────────────────────────
# Nodes
# ────────────────────────────────────────────────────────────────────────────

async def retrieve_node(state: GraphState) -> dict[str, Any]:
    """Run bge-m3 retrieval over the menu corpus using the user's message."""
    query = state["user_message"]
    retriever = get_retriever()
    chunks = await retriever.search(query)
    ctx = format_chunks_for_prompt(chunks)
    log.info("rag_retrieved", extra={"n_chunks": len(chunks), "query": query[:80]})
    return {"rag_context": ctx}


async def llm_node(state: GraphState) -> dict[str, Any]:
    """Call vLLM with current messages + tools.

    Returns the assistant message (text or tool_calls). The conditional
    edge below decides whether we loop into the tools node or finish.
    """
    messages = list(state["messages"])  # copy

    # Inject RAG context as an ephemeral system note BEFORE the latest user msg.
    # We don't persist this to Valkey — it's per-turn.
    rag_ctx = state.get("rag_context", "")
    if rag_ctx:
        # Find the last user message and insert RAG before it
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                messages.insert(i, {"role": "system", "content": rag_ctx})
                break

    resp = await chat_complete(messages, tools=TOOL_SCHEMAS, tool_choice="auto", stream=False)
    choice = resp.choices[0]
    msg = choice.message

    # Build the assistant message in OpenAI shape
    assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]

    log.info(
        "llm_turn",
        extra={
            "has_tool_calls": bool(msg.tool_calls),
            "n_tool_calls": len(msg.tool_calls or []),
            "content_len": len(msg.content or ""),
        },
    )

    update: dict[str, Any] = {"messages": [assistant_msg]}
    if not msg.tool_calls:
        update["final_text"] = msg.content or ""
    return update


async def tools_node(state: GraphState) -> dict[str, Any]:
    """Execute all tool calls from the most recent assistant message."""
    last_msg = state["messages"][-1]
    tool_calls = last_msg.get("tool_calls", [])

    tool_messages: list[dict[str, Any]] = []
    for tc in tool_calls:
        name = tc["function"]["name"]
        args = tc["function"]["arguments"]
        log.info("tool_invoke", extra={"tool": name, "tool_call_id": tc["id"]})
        result = await execute_tool(name, args)
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": name,
                "content": result,
            }
        )

    iters = state.get("tool_iterations", 0) + 1
    return {"messages": tool_messages, "tool_iterations": iters}


def should_continue(state: GraphState) -> str:
    """Conditional edge: after llm_node, route to tools or end."""
    last_msg = state["messages"][-1]
    has_tool_calls = bool(last_msg.get("tool_calls"))
    iters = state.get("tool_iterations", 0)
    max_iters = get_settings().max_tool_iterations

    if has_tool_calls and iters < max_iters:
        return "tools"
    if has_tool_calls and iters >= max_iters:
        log.warning("tool_loop_exhausted", extra={"iterations": iters})
        # Force-stop: replace the dangling tool-call message with a fallback
        # We can't mutate state here cleanly, so we let it fall through and
        # the main handler will detect missing final_text.
        return END
    return END


# ────────────────────────────────────────────────────────────────────────────
# Graph build
# ────────────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("llm", llm_node)
    g.add_node("tools", tools_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "llm")
    g.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "llm")

    return g.compile()


# Module-level compiled graph
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def build_initial_messages(history: list[dict[str, Any]], user_message: str) -> list[dict[str, Any]]:
    """Prepend system prompt, append the new user message."""
    msgs: list[dict[str, Any]] = [{"role": "system", "content": AKI_SYSTEM_PROMPT}]
    msgs.extend(history)
    msgs.append({"role": "user", "content": user_message})
    return msgs


async def run_turn(
    session_id: str,
    user_message: str,
    history: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Run a single conversational turn.

    Returns:
      (final_assistant_text, updated_history)
      where updated_history is what should be persisted to Valkey
      (excludes system + ephemeral RAG context).
    """
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

    final_text = final_state.get("final_text") or ""
    if not final_text:
        final_text = "Sorry, I had trouble completing that. Could you rephrase?"

    # Updated history = everything except the system prompt and ephemeral RAG
    # injection. We rebuild by keeping the original history plus only messages
    # added during this turn that are persistence-worthy.
    new_messages: list[dict[str, Any]] = []
    for m in final_state["messages"]:
        if m.get("role") == "system":
            continue  # drop system & ephemeral RAG
        new_messages.append(m)

    return final_text, new_messages
