"""LangGraph orchestration for Aki.

Flow:
                ┌──────────────┐
   start ───▶   │    intent    │  classify: order_food | check_order_status | general
                └──────┬───────┘
                       │
          ┌────────────┼────────────────┐
          │            │                │
     order_food  check_order_status  general
          │            │                │
          ▼            │                │
  ┌──────────────┐     │                │
  │collect_params│     │                │
  └──────┬───────┘     │                │
         │             │                │
  complete?            │                │
    ├─ no → END        │                │
    │  (ask user)      │                │
    └─ yes             │                │
         ▼             ▼                ▼
  ┌──────────────────────────────────────┐
  │             retrieve                 │  (bge-m3 over menu corpus)
  └──────────────────┬───────────────────┘
                     │
    order_food only  │
         ▼           │
  ┌──────────────┐   │
  │semantic_match│   │  (resolves names → real IDs via pgvector)
  └──────┬───────┘   │
         └───────────┘
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
               └─ no ──▶ END
"""
from __future__ import annotations

import json
from typing import Annotated, Any, Optional, TypedDict
import re

from langgraph.graph import StateGraph, END

from agent.app.config import get_settings
from agent.app.graph.intent import intent_node, collect_params_node
from agent.app.graph.prompts import AKI_SYSTEM_PROMPT
from agent.app.graph.semantic_match import semantic_match_node
from agent.app.llm import chat_complete
from agent.app.logging_setup import get_logger
from agent.app.rag import format_chunks_for_prompt, get_retriever
from agent.app.tools import TOOL_SCHEMAS, execute_tool

log = get_logger(__name__)


_AUTH_REQUIRED_TOOLS = frozenset({
    "place_order",
    "get_order_status",
    "get_active_orders",
    "cancel_order",
    "get_user_addresses",
})


def _merge_messages(left: list[dict], right: list[dict]) -> list[dict]:
    """Reducer: append new messages to existing history."""
    return left + right


class GraphState(TypedDict, total=False):
    session_id: str
    user_message: str
    messages: Annotated[list[dict[str, Any]], _merge_messages]
    rag_context: str
    jwt_token: str

    # Intent routing
    intent: str                          # "order_food" | "check_order_status" | "general"

    # Order parameter accumulation (persisted across turns via Valkey)
    order_params: dict[str, Any]         # {restaurant_name, items: [{name, quantity}]}
    params_complete: bool

    # Semantic match results
    semantic_matches: list[dict[str, Any]]

    # Loop guard + output
    tool_iterations: int
    final_text: str


# ────────────────────────────────────────────────────────────────────────────
# Nodes
# ────────────────────────────────────────────────────────────────────────────

_QUERY_EXTRACT_SYSTEM = """\
You are a structured data extractor for a food delivery app.
Extract from the user message:
- restaurant_name: name of a specific restaurant, or null
- menu_item: a specific food or drink item, or null
- cuisine: a broad cuisine category ONLY if explicitly stated
  (e.g. "punjabi", "chinese", "south indian"), or null

Rules:
- Extract ONLY what is literally present. Do not infer cuisine from dish names.
- "I want something punjabi" → cuisine: "punjabi", others null
- "biryani from Hotel XYZ" → restaurant_name: "Hotel XYZ", menu_item: "biryani", cuisine: null
- "something creamy" → all null
Return ONLY valid JSON, nothing else.
{"restaurant_name": "..or null", "menu_item": "..or null", "cuisine": "..or null"}
"""

async def _extract_query_fields(user_message: str) -> dict:
    resp = await chat_complete(
        [
            {"role": "system", "content": _QUERY_EXTRACT_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        tools=None,
        tool_choice=None,
        stream=False,
    )
    raw = (resp.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"restaurant_name": None, "menu_item": None, "cuisine": None}


async def retrieve_node(state: GraphState) -> dict[str, Any]:
    query = state["user_message"]

    # Extract structured fields from the query
    extracted = await _extract_query_fields(query)
    restaurant_name = extracted.get("restaurant_name")
    menu_item       = extracted.get("menu_item")
    cuisine         = extracted.get("cuisine")

    retriever = get_retriever()
    chunks = await retriever.search(
        query,
        restaurant_name=restaurant_name,
        menu_item=menu_item,
        cuisine=cuisine,
    )

    ctx = format_chunks_for_prompt(chunks)
    log.info("rag_retrieved", extra={
        "n_chunks": len(chunks),
        "query": query[:80],
        "extracted": extracted,
    })
    return {"rag_context": ctx}


async def llm_node(state: GraphState) -> dict[str, Any]:
    """Call vLLM with current messages + tools."""
    messages = list(state["messages"])

    # Inject RAG context as ephemeral system note before the last user message
    rag_ctx = state.get("rag_context", "")
    if rag_ctx:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                messages.insert(i, {"role": "system", "content": rag_ctx})
                break

    resp = await chat_complete(messages, tools=TOOL_SCHEMAS, tool_choice="auto", stream=False)
    choice = resp.choices[0]
    msg = choice.message

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
    jwt_token = state.get("jwt_token", "")

    tool_messages: list[dict[str, Any]] = []
    for tc in tool_calls:
        name = tc["function"]["name"]

        # Parse LLM-provided arguments
        raw_args = tc["function"]["arguments"]
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args = {}
        else:
            args = dict(raw_args)

        # Silently inject JWT for tools that require authentication
        if name in _AUTH_REQUIRED_TOOLS:
            args["jwt_token"] = jwt_token

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


# ────────────────────────────────────────────────────────────────────────────
# Routing functions
# ────────────────────────────────────────────────────────────────────────────

def route_intent(state: GraphState) -> str:
    """After intent_node: route to collect_params, retrieve, or retrieve directly."""
    intent = state.get("intent", "general")
    if intent == "order_food":
        return "collect_params"
    # check_order_status and general both go straight to retrieve
    return "retrieve"


def route_after_collect_params(state: GraphState) -> str:
    """After collect_params_node: go to semantic_match or end turn."""
    if state.get("params_complete"):
        return "semantic_match"
    # Params incomplete — final_text already set with clarifying question
    return END


def route_after_semantic_match(state: GraphState) -> str:
    """After semantic_match_node: always go to retrieve for RAG context."""
    return "retrieve"


def route_after_retrieve(state: GraphState) -> str:
    """After retrieve_node: always go to llm."""
    return "llm"


def should_continue(state: GraphState) -> str:
    """After llm_node: loop to tools or end."""
    last_msg = state["messages"][-1]
    has_tool_calls = bool(last_msg.get("tool_calls"))
    iters = state.get("tool_iterations", 0)
    max_iters = get_settings().max_tool_iterations

    if has_tool_calls and iters < max_iters:
        return "tools"
    if has_tool_calls and iters >= max_iters:
        log.warning("tool_loop_exhausted", extra={"iterations": iters})
        return END
    return END


# ────────────────────────────────────────────────────────────────────────────
# Graph build
# ────────────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(GraphState)

    # Register nodes
    g.add_node("intent", intent_node)
    g.add_node("collect_params", collect_params_node)
    g.add_node("semantic_match", semantic_match_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("llm", llm_node)
    g.add_node("tools", tools_node)

    # Entry point
    g.set_entry_point("llm")

    # Intent routing
    g.add_conditional_edges(
       "intent",
       route_intent,
       {
           "collect_params": "collect_params",
           "retrieve": "retrieve",
       },
    )

    # Param collection routing
    g.add_conditional_edges(
       "collect_params",
       route_after_collect_params,
       {
          "semantic_match": "semantic_match",
           END: END,
       },
    )

    # Semantic match always goes to retrieve
    g.add_edge("semantic_match", "retrieve")

    # Retrieve always goes to llm
    g.add_edge("retrieve", "llm")

    # LLM routing — tool loop or end
    g.add_conditional_edges(
        "llm",
        should_continue,
        {"tools": "tools", END: END},
    )

    # Tools always loop back to llm
    g.add_edge("tools", "llm")

    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def build_initial_messages(
    history: list[dict[str, Any]],
    user_message: str,
) -> list[dict[str, Any]]:
    """Prepend system prompt, append the new user message."""
    msgs: list[dict[str, Any]] = [{"role": "system", "content": AKI_SYSTEM_PROMPT}]
    msgs.extend(history)
    msgs.append({"role": "user", "content": user_message})
    return msgs


async def run_turn(
    session_id: str,
    user_message: str,
    history: list[dict[str, Any]],
    order_params: Optional[dict[str, Any]] = None,
    jwt_token: str = "",
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Run a single conversational turn.

    Args:
        session_id:    Unique session identifier.
        user_message:  Latest message from the user.
        history:       Prior conversation messages from Valkey.
        order_params:  Accumulated order parameters from Valkey.
        jwt_token:     JWT bearer token from the mobile app. Injected into
                       auth-required tool calls transparently.

    Returns:
        (final_assistant_text, updated_history, updated_order_params)
    """
    graph = get_graph()
    initial_messages = build_initial_messages(history, user_message)

    final_state = await graph.ainvoke(
        {
            "session_id": session_id,
            "user_message": user_message,
            "messages": initial_messages,
            "tool_iterations": 0,
            "order_params": order_params or {},
            "params_complete": False,
            "semantic_matches": [],
            "jwt_token":       jwt_token,
        }
    )

    final_text = final_state.get("final_text") or ""
    if not final_text:
        final_text = "Sorry, I had trouble completing that. Could you rephrase?"

    # Rebuild history — exclude system prompts and ephemeral RAG/semantic injections
    new_messages: list[dict[str, Any]] = []
    for m in final_state["messages"]:
        role = m.get("role")
        if role == "system":
            continue
        new_messages.append(m)

    # Return updated order_params — reset if order was placed
    updated_order_params = final_state.get("order_params") or {}
    # Clear params if order was successfully placed (place_order tool was called)
    for m in new_messages:
        if m.get("role") == "tool" and m.get("name") == "place_order":
            try:
                result = json.loads(m.get("content", "{}"))
                if result.get("success"):
                    updated_order_params = {}
            except (json.JSONDecodeError, AttributeError):
                pass

    return final_text, new_messages, updated_order_params