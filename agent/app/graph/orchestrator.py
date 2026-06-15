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
    └─ yes             ▼                ▼
                ┌──────────────────────────────────────┐
                │             retrieve                  │
                │  bge-m3 over menu corpus              │
                │  uses order_params filters when set   │
                └──────────────────┬───────────────────┘
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

Message hygiene contract
────────────────────────
state["messages"] contains ONLY:
  • system prompt (position 0)
  • prior user/assistant/tool turns from Valkey history
  • the current user message
  • assistant + tool messages produced during the llm↔tools loop

It NEVER contains:
  • intent classifier output
  • collect_params clarifying questions   ← these go to final_text only
  • ephemeral RAG context injections      ← local copy in llm_node only
"""
from __future__ import annotations

import json
import re
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import StateGraph, END

from agent.app.config import get_settings
from agent.app.graph.intent import intent_node, collect_params_node
from agent.app.graph.prompts import AKI_SYSTEM_PROMPT
from agent.app.llm import chat_complete
from agent.app.logging_setup import get_logger
from agent.app.rag.retriever import format_chunks_for_prompt, get_retriever
from agent.app.tools import TOOL_SCHEMAS, execute_tool

log = get_logger(__name__)


_AUTH_REQUIRED_TOOLS = frozenset({
    "place_order",
    "get_order_status",
    "get_active_orders",
    "cancel_order",
    "get_user_addresses",
})

# Tools that END a flow — after a successful one, wipe order_params + active_intent
# so the next turn starts clean and reclassifies fresh.
# NOTE: get_active_orders is NOT here — it's a step toward cancel, not terminal.
TERMINAL_TOOLS = {"place_order", "cancel_order", "get_order_status","discard_current_order"}

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
    intent: str                  # "order_food" | "check_order_status" | "cancel_order" | "general"

    # Order parameter accumulation (persisted across turns via Valkey)
    order_params: dict[str, Any]
    params_complete: bool

    # Set True when all order details confirmed — forces place_order tool call
    order_ready: bool

    # Loop guard + output
    tool_iterations: int
    final_text: str

    # Set by retrieve_node when a named restaurant couldn't be matched
    restaurant_not_found: bool

    # Set by retrieve_node when the requested item isn't available at the
    # restaurant. run_turn() persists this into order_params so it survives
    # the Valkey round-trip. collect_params_node clears items[] when it sees
    # this flag, so stale items don't accumulate when the user picks a new one.
    item_rejected: bool

    # Set by llm_node when restaurant + items + payment are all confirmed.
    # Frontend intercepts final_text "CHECKOUT_READY::{...}" prefix and
    # navigates to the checkout screen. order_params is reset after this.
    checkout_ready: bool

    # Set by llm_node when user explicitly confirms the order summary.
    # Persisted in order_params so it survives the Valkey round-trip.
    # Triggers checkout handoff on the next turn.
    items_confirmed: bool


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
    """Menu retrieval using static rules (exact → contains → trigram).

    The retriever handles all matching logic internally. This node is
    responsible for:
      - Resolving restaurant_name and menu_item from order_params or LLM extraction
      - Interpreting empty results as not-found and short-circuiting to END
      - Presenting restaurant choices when item is known but restaurant isn't
      - Passing confirmed chunks to LLM as RAG context

    item_rejected flag:
      Set to True when the requested item isn't available at the restaurant.
      run_turn() persists this into order_params (Valkey) so collect_params_node
      sees it next turn and clears stale items before merging the user's new pick.
      Set to False on all other paths so the flag is always explicitly returned.
    """
    query = state["user_message"]
    order_params = state.get("order_params") or {}
    retriever = get_retriever()

    # Initialize so pure-discovery path never hits NameError
    matched_restaurant_id   = None
    matched_restaurant_name = None
    chunks = []
    cuisine = None
    menu_item = None
    restaurant_name = None

    # ── Resolve names ─────────────────────────────────────────────────────────
    if order_params.get("restaurant_name") or order_params.get("items"):
        restaurant_name = order_params.get("restaurant_name")
        items = order_params.get("items") or []
        menu_item = items[0]["name"] if items else None
        cuisine = None
        log.info("retrieve_using_order_params", extra={
            "restaurant": restaurant_name,
            "menu_item": menu_item,
        })
    else:
        extracted = await _extract_query_fields(query)
        restaurant_name = extracted.get("restaurant_name")
        menu_item       = extracted.get("menu_item")
        cuisine         = extracted.get("cuisine")
        log.info("retrieve_using_extracted_fields", extra={
            "extracted": extracted,
            "query": query[:80],
        })

    # ── Restaurant named — confirm it exists, then find item ─────────────────
    if restaurant_name:
        # Step 1: confirm restaurant exists (name-only search returns menu sample)
        restaurant_check = await retriever.search(
            restaurant_name,
            restaurant_name=restaurant_name,
            menu_item=None,
            cuisine=None,
            top_k=1,
        )

        if not restaurant_check:
            msg = (
                f"Sorry, I couldn't find a restaurant called '{restaurant_name}' "
                f"on EasyCater. Would you like to search for something else?"
            )
            log.info("restaurant_not_found", extra={"restaurant_name": restaurant_name})
            return {
                "rag_context": "",
                "final_text": msg,
                "restaurant_not_found": True,
                "item_rejected": False,
            }

        matched_restaurant_id   = restaurant_check[0].restaurant_id
        matched_restaurant_name = restaurant_check[0].restaurant_name
        log.info("restaurant_confirmed", extra={
            "input": restaurant_name,
            "matched": matched_restaurant_name,
            "id": matched_restaurant_id,
        })

        if not menu_item:
            # No item requested — return menu sample as RAG context
            chunks = restaurant_check
        else:
            # Step 2: find the item within the confirmed restaurant
            chunks = await retriever.search(
                menu_item,
                restaurant_name=matched_restaurant_name,
                menu_item=menu_item,
                cuisine=cuisine,
                restaurant_id=matched_restaurant_id,
            )

            if not chunks:
                # Restaurant exists but item not on menu — fetch a sample to suggest
                sample = await retriever.search(
                    restaurant_name,
                    restaurant_name=matched_restaurant_name,
                    menu_item=None,
                    cuisine=None,
                    top_k=3,
                    restaurant_id=matched_restaurant_id,
                )
                suggestion_str = ", ".join(
                    f"{c.item_name} (\u20b9{c.price:.0f})" for c in sample
                ) if sample else "other items"
                msg = (
                    f"Sorry, '{menu_item}' isn't available at {matched_restaurant_name}. "
                    f"They have: {suggestion_str}. "
                    f"Would you like one of these, or should I search for "
                    f"'{menu_item}' at another restaurant?"
                )
                log.info("item_not_found_at_restaurant", extra={
                    "restaurant": matched_restaurant_name,
                    "restaurant_id": matched_restaurant_id,
                    "menu_item": menu_item,
                })
                return {
                    "rag_context": "",
                    "final_text": msg,
                    "restaurant_not_found": False,
                    "item_rejected": True,   # ← tells collect_params to clear items next turn
                }

    # ── Item known, no restaurant — find which restaurants serve it ───────────
    elif menu_item:
        chunks = await retriever.search(
            query,
            menu_item=menu_item,
            cuisine=cuisine,
            top_k=10,
        )

        if not chunks:
            msg = (
                f"Sorry, I couldn't find '{menu_item}' at any restaurant on EasyCater. "
                f"Would you like to try a different dish or browse what's available?"
            )
            log.info("item_not_found_anywhere", extra={"menu_item": menu_item})
            return {
                "rag_context": "",
                "final_text": msg,
                "restaurant_not_found": False,
                "item_rejected": True,   # ← user will pick a different item next turn
            }

        # Deduplicate by restaurant — keep one entry per restaurant
        seen: set[str] = set()
        unique_restaurants: list[dict] = []
        for chunk in chunks:
            if chunk.restaurant_id not in seen:
                seen.add(chunk.restaurant_id)
                unique_restaurants.append({
                    "name": chunk.restaurant_name,
                    "id": chunk.restaurant_id,
                    "price": chunk.price,
                    "rating": chunk.rating,
                })

        lines = [f"Here are the restaurants serving '{menu_item}' on EasyCater:\n"]
        for i, r in enumerate(unique_restaurants, 1):
            rating_str = f" \u2b50{r['rating']:.1f}" if r["rating"] else ""
            price_str  = f" \u00b7 \u20b9{r['price']:.0f}" if r["price"] else ""
            lines.append(f"{i}. {r['name']}{rating_str}{price_str}")
        lines.append("\nWhich restaurant would you like to order from?")

        msg = "\n".join(lines)
        log.info("restaurant_choice_presented", extra={
            "menu_item": menu_item,
            "n_restaurants": len(unique_restaurants),
        })
        return {
            "rag_context": "",
            "final_text": msg,
            "restaurant_not_found": False,
            "item_rejected": False,   # item exists, just picking a restaurant
        }

    # ── Pure discovery — no restaurant or item named ──────────────────────────
    else:
        chunks = await retriever.search(query, cuisine=cuisine)

    if matched_restaurant_id and not chunks:
        item_label = f"'{menu_item}'" if menu_item else "that item"
        msg = (
            f"Sorry, {item_label} doesn't seem to be available at "
            f"{matched_restaurant_name}. Would you like to see what's on their menu, "
            f"or search for {item_label} at another restaurant?"
        )
        log.info("item_not_found_at_restaurant", extra={
            "restaurant_id": matched_restaurant_id,
            "menu_item": menu_item,
        })
        return {
            "rag_context": "",
            "final_text": msg,
            "restaurant_not_found": False,
            "item_rejected": True,
        }

    ctx = format_chunks_for_prompt(chunks)
    log.info("rag_retrieved", extra={"n_chunks": len(chunks), "query": query[:80]})
    return {
        "rag_context": ctx,
        "restaurant_not_found": False,
        "item_rejected": False,
    }


async def llm_node(state: GraphState) -> dict[str, Any]:
    """Call vLLM with current messages + tools.

    RAG context is injected into a LOCAL copy of messages for this call only —
    never written back to state["messages"].

    When restaurant and items are confirmed, short-circuits to END and emits
    a CHECKOUT_READY signal instead of calling vLLM. The frontend intercepts
    this and navigates to the checkout screen.
    """
    messages = list(state["messages"])

    rag_ctx = state.get("rag_context", "")
    if rag_ctx:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                messages.insert(i, {"role": "system", "content": f"[CONTEXT]\n{rag_ctx}\n[/CONTEXT]"})
                break

    # ── Detect if all order details are confirmed → emit checkout signal ───────
    # Two-stage gate:
    #   Stage 1 (items_confirmed=False): LLM confirms quantity + special
    #     instructions with the user. Once user says yes/confirmed, collect_params
    #     sets items_confirmed=True in order_params.
    #   Stage 2 (items_confirmed=True): short-circuit to checkout immediately.
    order_params = state.get("order_params") or {}
    items_confirmed = order_params.get("items_confirmed", False)

    all_confirmed = (
        bool(order_params.get("restaurant_name"))
        and bool(order_params.get("items"))
        and items_confirmed
    )

    # ── Inject order status block so LLM knows what's confirmed vs missing ────
    order_status_lines = []
    if order_params.get("restaurant_name") and order_params.get("items"):
        items_list = order_params.get("items") or []
        item_str = ", ".join(
            f"{it['quantity']}x {it['name']}"
            + (f" ({it['special_instructions']})" if it.get("special_instructions") else "")
            for it in items_list
        )
        order_status_lines.append(
            f"CURRENT ORDER (IN PROGRESS — NOT YET PLACED, no order_id): "
            f"{item_str} from {order_params['restaurant_name']}"
        )

        if all_confirmed:
            order_status_lines.append(
                "STATUS: ✅ ALL CONFIRMED — tell the user you are taking them to checkout. Do NOT call any tool."
            )
        elif not items_confirmed:
            order_status_lines.append(
                "STATUS: ⏳ PENDING CONFIRMATION — confirm the exact items, quantities, "
                "and any special instructions with the user in plain language. "
                "Once the user says yes/confirmed, you MUST reply with the word CONFIRMED on its own line."
            )

        order_status = "\n".join(order_status_lines)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                messages.insert(i, {"role": "system", "content": f"[ORDER STATUS]\n{order_status}\n[/ORDER STATUS]"})
                break

    # ── Short-circuit to checkout — skip LLM call entirely ───────────────────
    if all_confirmed:
        items_list = order_params.get("items") or []
        item_str = ", ".join(
            f"{it['quantity']}x {it['name']}" for it in items_list
        )
        checkout_text = (
            f"Great! Taking you to checkout for {item_str} "
            f"from {order_params['restaurant_name']}. "
            f"You can choose delivery or pickup and confirm your address there."
        )
        checkout_payload = json.dumps({
            "restaurant_id": None,
            "items": items_list,
        })
        log.info("checkout_ready", extra={
            "restaurant": order_params.get("restaurant_name"),
            "items":      item_str,
        })
        return {
            "messages":       [{"role": "assistant", "content": f"CHECKOUT_READY::{checkout_payload}"}],
            "final_text":     checkout_text,
            "checkout_ready": True,
        }

    tool_choice = "auto"
    log.info("llm_auto_mode", extra={
        "restaurant": order_params.get("restaurant_name"),
        "missing":    order_status_lines[-1] if order_status_lines else "no order in progress",
    })

    resp = await chat_complete(
        messages,
        tools=TOOL_SCHEMAS,
        tool_choice=tool_choice,
        stream=False,
    )
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
            "forced": all_confirmed,
        },
    )

    update: dict[str, Any] = {"messages": [assistant_msg]}
    if not msg.tool_calls:
        content = msg.content or ""
        update["final_text"] = content

        # Detect confirmation signal — LLM says CONFIRMED on its own line
        # when the user has acknowledged the order summary. Set items_confirmed
        # in order_params so next turn short-circuits to checkout.
        if "CONFIRMED" in [line.strip() for line in content.splitlines()]:
            updated_params = dict(state.get("order_params") or {})
            updated_params["items_confirmed"] = True
            update["order_params"] = updated_params
            # Strip the CONFIRMED line from what the user sees
            visible_lines = [l for l in content.splitlines() if l.strip() != "CONFIRMED"]
            update["final_text"] = "\n".join(visible_lines).strip()
            log.info("items_confirmed_set", extra={
                "restaurant": updated_params.get("restaurant_name"),
            })

    return update


async def tools_node(state: GraphState) -> dict[str, Any]:
    """Execute all tool calls from the most recent assistant message."""
    last_msg = state["messages"][-1]
    tool_calls = last_msg.get("tool_calls", [])
    jwt_token = state.get("jwt_token", "")

    log.info("tools_node_jwt_check", extra={
        "jwt_present": bool(jwt_token),
        "jwt_prefix":  jwt_token[:20] if jwt_token else "EMPTY",
        "n_tool_calls": len(tool_calls),
    })

    tool_messages: list[dict[str, Any]] = []
    updated_order_params = dict(state.get("order_params") or {})

    for tc in tool_calls:
        name = tc["function"]["name"]

        raw_args = tc["function"]["arguments"]
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args = {}
        else:
            args = dict(raw_args)

        if name in _AUTH_REQUIRED_TOOLS:
            args["jwt_token"] = jwt_token

        log.info("tool_invoke", extra={"tool": name, "tool_call_id": tc["id"]})
        result = await execute_tool(name, args)

        if name == "place_order":
            try:
                place_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                if place_args.get("delivery_address"):
                    updated_order_params["delivery_address"] = place_args["delivery_address"]
            except (json.JSONDecodeError, AttributeError):
                pass

        if name == "get_user_addresses":
            try:
                addr_result = json.loads(result)
                log.info("addresses_fetched", extra={
                    "success": addr_result.get("success"),
                    "count":   addr_result.get("total", 0),
                })
            except (json.JSONDecodeError, AttributeError):
                pass

        tool_messages.append({
            "role":         "tool",
            "tool_call_id": tc["id"],
            "name":         name,
            "content":      result,
        })

    iters = state.get("tool_iterations", 0) + 1
    return {
        "messages":        tool_messages,
        "tool_iterations": iters,
        "order_params":    updated_order_params,
    }


# ────────────────────────────────────────────────────────────────────────────
# Routing functions
# ────────────────────────────────────────────────────────────────────────────

def route_intent(state: GraphState) -> str:
    intent = state.get("intent", "general")
    if intent == "order_food":
        return "collect_params"
    if intent in ("cancel_order", "check_order_status"):
        return "llm"          # get_active_orders / cancel_order / get_order_status handle it
    return "retrieve"          # general / discovery only


def route_after_collect_params(state: GraphState) -> str:
    if state.get("params_complete"):
        return "retrieve"
    return END


def route_after_retrieve(state: GraphState) -> str:
    if state.get("restaurant_not_found") or (
        state.get("final_text") and not state.get("rag_context")
    ):
        return END
    return "llm"


def should_continue(state: GraphState) -> str:
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
    log.info("graph_building", extra={"entry": "intent", "nodes": "full_pipeline"})
    g = StateGraph(GraphState)

    g.add_node("intent", intent_node)
    g.add_node("collect_params", collect_params_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("llm", llm_node)
    g.add_node("tools", tools_node)

    g.set_entry_point("intent")

    g.add_conditional_edges(
        "intent",
        route_intent,
        {"collect_params": "collect_params", "retrieve": "retrieve", "llm":"llm",}
    )

    g.add_conditional_edges(
        "collect_params",
        route_after_collect_params,
        {"retrieve": "retrieve", END: END},
    )

    g.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"llm": "llm", END: END},
    )

    g.add_conditional_edges(
        "llm",
        should_continue,
        {"tools": "tools", END: END},
    )

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

    Returns:
        (final_assistant_text, updated_history, updated_order_params)

    History write rules
    ───────────────────
    • Normal LLM turns: assistant + tool messages appended (system stripped).
    • collect_params incomplete: clarifying question appended manually.
    • retrieve short-circuit: not-found message appended manually.
    • RAG injections: never written to state["messages"].

    item_rejected persistence
    ─────────────────────────
    When retrieve_node sets item_rejected=True, run_turn writes it into
    updated_order_params so it survives the Valkey round-trip. On the next
    turn collect_params_node reads it from order_params, clears items[], and
    pops the flag before merging the user's new item selection.
    """
    graph = get_graph()
    initial_messages = build_initial_messages(history, user_message)

    final_state = await graph.ainvoke(
        {
            "session_id":           session_id,
            "user_message":         user_message,
            "messages":             initial_messages,
            "tool_iterations":      0,
            "order_params":         order_params or {},
            "params_complete":      False,
            "jwt_token":            jwt_token,
            "restaurant_not_found": False,
            "item_rejected":        False,
            "checkout_ready":       False,
        }
    )

    final_text = final_state.get("final_text") or ""
    if not final_text:
        final_text = "Sorry, I had trouble completing that. Could you rephrase?"

    # ── Rebuild Valkey history ────────────────────────────────────────────────
    new_messages: list[dict[str, Any]] = []
    for m in final_state["messages"]:
        # Strip system messages — includes the system prompt, ephemeral RAG
        # context injections, and ORDER STATUS blocks, all of which are
        # role="system" and must never persist to Valkey.
        if m.get("role") == "system":
            continue
        new_messages.append(m)

    intent = final_state.get("intent", "general")
    params_complete_flag = final_state.get("params_complete", False)
    tool_ran = any(m.get("role") == "tool" for m in new_messages)
    retrieve_short_circuited = bool(
        final_state.get("restaurant_not_found")
        or (
            final_state.get("final_text")
            and not final_state.get("rag_context")
            and not tool_ran
            and intent == "order_food"
            and params_complete_flag
        )
    )

    if intent == "order_food" and not params_complete_flag and not tool_ran and final_text:
        new_messages.append({"role": "assistant", "content": final_text})
    elif retrieve_short_circuited and final_text:
        new_messages.append({"role": "assistant", "content": final_text})

    # ── Reset order_params on checkout handoff or successful place_order ────────
    updated_order_params = final_state.get("order_params") or {}

    # Checkout handoff — frontend takes over, clear order state
    if final_state.get("checkout_ready"):
        updated_order_params = {}
        log.info("order_params_reset", extra={"reason": "checkout_ready"})

   
    # ── Terminal tools — reset order_params (incl. active_intent) ─────────────
    # A successful place_order / cancel_order / get_order_status ends the flow.
    # Wiping order_params clears active_intent too, so the next turn starts
    # clean and reclassifies fresh. A tool that explicitly reports success=False
    # (e.g. a cancel that failed) does NOT reset, so the user stays in the flow
    # to retry.
    for m in new_messages:
        if m.get("role") == "tool" and m.get("name") in TERMINAL_TOOLS:
            try:
                result = json.loads(m.get("content", "{}"))
            except (json.JSONDecodeError, AttributeError):
                result = {}
            if result.get("success") is not False:
                updated_order_params = {}
                log.info("order_params_reset", extra={
                    "reason": f"terminal_tool:{m.get('name')}",
                })
                break

    # ── Persist item_rejected into order_params for next turn ─────────────────
    # collect_params_node reads this from order_params (not GraphState) because
    # GraphState doesn't survive between turns — only Valkey does.
    # The flag is consumed by collect_params_node on the next turn (popped after
    # clearing items[]), so it only fires once.
    updated_order_params["item_rejected"] = final_state.get("item_rejected", False)

    return final_text, new_messages, updated_order_params