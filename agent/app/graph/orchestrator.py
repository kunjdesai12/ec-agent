"""Agentic tool-calling orchestrator for Aki.

New architecture (replaces the LangGraph node pipeline):

    user query ──▶ LLM ──▶ tool call(s) ──▶ tool results ──▶ LLM ──▶ reply
                    ▲                                          │
                    └──────────── (loop until no tool call) ───┘

No retriever node. No intent classifier. The retriever is exposed purely as
tools (search_food, confirm_restaurant) via registry.py and is called by the
LLM only when it decides it needs food data — there is no pre-injected
[CONTEXT] RAG block anymore.

Per turn:
  1. Load the rolling message history for this session (in-memory store).
  2. Append the user message (and an optional developer block for the order
     scratchpad / [ORDER STATUS], built by the caller from order_params).
  3. Loop: call the LLM with the tool schemas. If it emits tool calls, run them
     (injecting session-scoped args like jwt_token), feed the results back, and
     call again. Stop when the LLM returns a plain assistant message.
  4. Persist the user turn + the final assistant turn, and return the text.

The intermediate tool-call rounds are kept in-context for the duration of the
turn but are NOT persisted across turns (keeps history small; the order
scratchpad carries cart state between turns, not the message log).
"""
from __future__ import annotations

import json
from typing import Any, Optional

from openai import AsyncOpenAI

from agent.app.config import get_settings
from agent.app.logging_setup import get_logger
from agent.app.graph.prompts import AKI_SYSTEM_PROMPT
from agent.app.tools.registry import TOOL_SCHEMAS, execute_tool
from agent.app import cart as cart_store

log = get_logger(__name__)  

_settings = get_settings()

# ── Config — map these to your actual settings fields ───────────────────────
# vLLM is OpenAI-compatible. Start it with auto tool-choice + the hermes parser
# so tool calls come back parsed:
#   vllm serve <model> --enable-auto-tool-choice --tool-call-parser hermes
VLLM_BASE_URL = getattr(_settings, "vllm_base_url", "http://localhost:8001/v1")
MODEL_NAME    = getattr(_settings, "model_name", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
TEMPERATURE   = getattr(_settings, "llm_temperature", 0.3)
MAX_TOKENS    = getattr(_settings, "llm_max_tokens", 1024)

# Safety valve: a single user turn may not chain more than this many tool
# rounds. Prevents a confused model from looping forever.
MAX_TOOL_ITERATIONS = 20

# Tools that finalize/clear the in-progress order. After one of these returns
# success, the caller should clear the order scratchpad (order_params).
TERMINAL_TOOLS = {"place_order", "cancel_order", "discard_current_order"}

# History kept per session. Keep the last N messages to bound context.
MAX_HISTORY_MESSAGES = 100                 # ~10 turns

_client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")


# ── Session history (in-memory) ─────────────────────────────────────────────
# Process-local message store, matching the current setup. History lives only
# for the lifetime of this process and is NOT shared across workers/replicas — a
# restart clears it, and last-write-wins on concurrent turns for the same
# session. If you later move to a shared store (Valkey, etc.), only these two
# functions need to change; run_turn is agnostic.
#
# If you already have an in-memory store object, point load_history /
# save_history at it instead of the module-level dict below.

_SESSION_STORE: dict[str, list[dict[str, Any]]] = {}


async def load_history(session_id: str) -> list[dict[str, Any]]:
    # Return a copy so callers can't mutate stored state in place.
    return list(_SESSION_STORE.get(session_id, []))


async def save_history(session_id: str, messages: list[dict[str, Any]]) -> None:
    _SESSION_STORE[session_id] = messages[-MAX_HISTORY_MESSAGES:]


def clear_history(session_id: str) -> None:
    """Drop a session's history (e.g. on logout or an explicit 'start over')."""
    _SESSION_STORE.pop(session_id, None)


# ── Tool execution with session-scoped arg injection ────────────────────────

def _inject_session_args(
    tool_args: dict[str, Any],
    *,
    session_id: str,
    jwt_token: str,
    user_lat: Optional[float],
    user_lon: Optional[float],
) -> dict[str, Any]:
    """Add args the LLM must never supply itself (session, auth, user geo).

    Handlers read these via args.get(...), so injecting them on every call is
    harmless for tools that don't use them. session_id is what lets the cart
    tools (add_to_cart / update_cart_item / remove_from_cart) and
    discard_current_order address the right session's scratchpad.
    """
    args = dict(tool_args)
    args["session_id"] = session_id
    args["jwt_token"] = jwt_token
    if user_lat is not None:
        args.setdefault("user_lat", user_lat)
    if user_lon is not None:
        args.setdefault("user_lon", user_lon)
    return args


# ── Main turn ───────────────────────────────────────────────────────────────

async def run_turn(
    session_id: str,
    user_message: str,
    *,
    jwt_token: str = "",
    system_prompt: str = AKI_SYSTEM_PROMPT,
    developer_block: Optional[str] = None,
    user_lat: Optional[float] = None,
    user_lon: Optional[float] = None,
) -> dict[str, Any]:
    """Run one user turn through the tool-calling loop.

    Args:
        session_id:      session key for the history store.
        user_message:    the user's text for this turn.
        jwt_token:       bearer token, injected into every tool call.
        system_prompt:   AKI_SYSTEM_PROMPT (tool-era version — see notes).
        developer_block: optional per-turn context to steer the model, e.g. the
                         [ORDER STATUS] scratchpad built from order_params. Sent
                         as a system message *after* the main prompt so it's
                         fresh each turn and never persisted.
        user_lat/lon:    optional user location for geo-aware tools.

    Returns:
        {
          "reply":          str,            # assistant text for the user/TTS
          "terminal_tool":  str | None,     # set if a TERMINAL_TOOL succeeded
          "tool_calls":     [str, ...],     # names called this turn (for logs)
        }
    """
    history = await load_history(session_id)

    # Build the working message list for this turn.
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)

    # Inject the live cart as the [ORDER STATUS] block so the model always
    # reasons against the real scratchpad, not its own memory. Rebuilt fresh
    # every turn from the cart store; never persisted into history.
    order_status = cart_store.format_for_prompt(session_id)
    if order_status:
        messages.append({"role": "system", "content": order_status})

    if developer_block:
        messages.append({"role": "system", "content": developer_block})
    messages.append({"role": "user", "content": user_message})

    called_tools: list[str] = []
    terminal_tool: Optional[str] = None
    final_text = ""

    #log.info("turn_start", extra={"session": session_id, "msgs": messages})

    for iteration in range(MAX_TOOL_ITERATIONS):
        # On the final allowed iteration, drop tools so the model is forced to
        # answer in words instead of requesting yet another call.
        use_tools = iteration < MAX_TOOL_ITERATIONS - 1
        
        log.info("iterations", extra={"session": session_id, "msgs": messages})

        resp = await _client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOL_SCHEMAS if use_tools else None,
            tool_choice="auto" if use_tools else None,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        msg = resp.choices[0].message
       
       
        # No tool calls → this is the user-facing reply. Done.
        if not getattr(msg, "tool_calls", None):
            final_text = (msg.content or "").strip()
            break

        # Record the assistant tool-call message before appending results
        # (OpenAI message ordering requires this).
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })
        # Execute each requested tool and append its result as a new message, to history.
        history.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        # Execute each requested tool and append its result.
        for tc in msg.tool_calls:
            name = tc.function.name
            called_tools.append(name)
            try:
                raw_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                raw_args = {}

            args = _inject_session_args(
                raw_args,
                session_id=session_id,
                jwt_token=jwt_token,
                user_lat=user_lat,
                user_lon=user_lon,
            )

            result_json = await execute_tool(name, args)  # returns JSON string

            # Track terminal-tool success so the caller can clear the scratchpad.
            if name in TERMINAL_TOOLS:
                try:
                    if json.loads(result_json).get("success"):
                        terminal_tool = name
                except json.JSONDecodeError:
                    pass

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_json,
            })

            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_json,
            })

        log.info("tool_round", extra={
            "session": session_id, "iteration": iteration,
            "tools": [tc.function.name for tc in msg.tool_calls],
        })
    else:
        # Loop exhausted without a plain reply.
        final_text = (
            "Sorry, I got a bit tangled up there. Could you tell me again what "
            "you'd like to order?"
        )
        log.warning("tool_loop_exhausted", extra={"session": session_id})

    # Persist only the user turn and the final assistant text (not the
    # intermediate tool rounds) to keep history small and replayable.
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": final_text})
    await save_history(session_id, history)

    return {
        "reply": final_text,
        "terminal_tool": terminal_tool,
        "tool_calls": called_tools,
    }