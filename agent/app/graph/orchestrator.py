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
  3. Optionally inject a pre-call REMINDER when session state suggests the
     next action MUST be a specific tool call (e.g. user just confirmed and
     cart is built → place_order). Never persisted.
  4. Loop: call the LLM with the tool schemas. If it emits tool calls, run them
     (injecting session-scoped args like jwt_token), feed the results back, and
     call again. Stop when the LLM returns a plain assistant message.
  5. VALIDATE the final assistant text against successful tool calls from this
     turn. If it claims an action (placed/cancelled/added/etc.) that no tool
     backed, inject a corrective system message and give the model ONE retry.
     After that, ship a safe fallback rather than a fabricated confirmation.
  6. Persist the user turn + the final assistant turn, and return the text.

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

# Safety layers. Adjust the module path to wherever you drop these two files.
from agent.app.safety.reminders import maybe_inject_reminder
from agent.app.safety.validator import validate_assistant_turn

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

# Post-generation validation retry budget. One retry is enough in practice;
# more risks cascading corrections that never converge.
MAX_VALIDATION_RETRIES = 1

# Tools that finalize/clear the in-progress order. After one of these returns
# success, the caller should clear the order scratchpad (order_params).
TERMINAL_TOOLS = {"place_order", "cancel_order", "discard_current_order"}

# History kept per session. Keep the last N messages to bound context.
MAX_HISTORY_MESSAGES = 100                 # ~10 turns

# What we ship when validation exhausts its retry budget. Deliberately vague:
# better to ask the user to repeat than to send a fabricated confirmation.
_SAFE_FALLBACK = (
    "Sorry, I hit a snag confirming that just now. Could you tell me once "
    "more what you'd like me to do?"
)

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


def _tool_succeeded(result_json: str) -> bool:
    """Best-effort parse of the tool result's success flag.

    Tools return JSON strings; a truthy `success` field means the action
    happened. Anything unparseable is treated as failure so the validator
    won't get spurious credit for a broken tool.
    """
    try:
        return bool(json.loads(result_json).get("success"))
    except (json.JSONDecodeError, AttributeError, TypeError):
        return False


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

    History strategy:
        We persist everything the LLM sees within a turn EXCEPT the injected
        system messages (order_status, developer_block, reminder, and any
        validator reprompt) which are rebuilt fresh every turn. This means
        history contains:

            user
            assistant (tool_calls)     ← Aki decided to call a tool
            tool (result)              ← what the tool returned
            tool (result)              ← second tool in same round if any
            assistant (tool_calls)     ← Aki called another tool
            tool (result)
            assistant (final text)     ← Aki's final reply to the user

        This gives the LLM full reasoning context across turns — it knows which
        restaurants/items it already resolved and won't re-call tools for them.
    """
    history = await load_history(session_id)

    # ── Build working message list for this turn ──────────────────────────────
    # Start with system prompt + full persisted history
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)

    # Inject live cart as [ORDER STATUS] block — rebuilt fresh every turn,
    # never persisted into history.
    order_status = cart_store.format_for_prompt(session_id)
    if order_status:
        messages.append({"role": "system", "content": order_status})

    if developer_block:
        messages.append({"role": "system", "content": developer_block})

    # Append this turn's user message to both working list and history
    user_msg = {"role": "user", "content": user_message}
    messages.append(user_msg)
    history.append(user_msg)

    # ── Pre-call reminder (safety layer 1) ────────────────────────────────────
    # If the user just confirmed and cart is built (or they just confirmed
    # cancellation the assistant asked about), nudge the model that its next
    # action MUST be the corresponding tool call — not a text confirmation.
    # Appended to `messages` only; never persisted.
    reminder = maybe_inject_reminder(
        user_message=user_message,
        history=history,
        cart_summary=order_status or "",
    )
    if reminder:
        log.info("aki.reminder.injected", extra={
            "session": session_id,
            "preview": reminder[:60],
        })
        messages.append({"role": "system", "content": reminder})

    # ── Tool-calling loop ─────────────────────────────────────────────────────
    called_tools: list[str] = []
    successful_tool_calls: list[str] = []   # for the validator
    terminal_tool: Optional[str] = None
    validation_retries_left = MAX_VALIDATION_RETRIES
    final_text = ""

    for iteration in range(MAX_TOOL_ITERATIONS):
        # On the final allowed iteration, drop tools so the model is forced
        # to produce a text reply instead of another tool call.
        use_tools = iteration < MAX_TOOL_ITERATIONS - 1

        log.info("turn_iteration", extra={
            "session": session_id,
            "iteration": iteration,
            "use_tools": use_tools,
        })

        resp = await _client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOL_SCHEMAS if use_tools else None,
            tool_choice="auto" if use_tools else None,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        msg = resp.choices[0].message

        # ── No tool calls → candidate final reply. VALIDATE before shipping. ──
        if not getattr(msg, "tool_calls", None):
            candidate_text = (msg.content or "").strip()

            # Safety layer 2: cross-check the text against successful tools.
            v = validate_assistant_turn(candidate_text, successful_tool_calls)
            if v.ok:
                final_text = candidate_text
                history.append({"role": "assistant", "content": final_text})
                break

            # Validation failed — the model claimed something no tool backed.
            log.warning("aki.validation.failed", extra={
                "session": session_id,
                "iteration": iteration,
                "violations": [x.value for x in v.violations],
                "successful_tool_calls": successful_tool_calls,
                "draft_preview": candidate_text[:200],
            })

            if validation_retries_left <= 0:
                # Fail closed: never ship the fabricated confirmation.
                log.error("aki.validation.exhausted", extra={
                    "session": session_id,
                    "draft_preview": candidate_text[:200],
                })
                final_text = _SAFE_FALLBACK
                history.append({"role": "assistant", "content": final_text})
                break

            validation_retries_left -= 1

            # Give the model its own bad draft + a corrective system message
            # so it can concretely see what to fix. Neither goes into `history`
            # — we don't want a fabricated confirmation persisted into future
            # turns even after correction.
            messages.append({"role": "assistant", "content": candidate_text})
            messages.append({"role": "system", "content": v.reprompt})
            continue  # let the loop run another model round

        # ── Tool calls requested ──────────────────────────────────────────────
        # Build the assistant message with tool_calls
        assistant_msg = {
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
        }
        # Append to both working list and history
        messages.append(assistant_msg)
        history.append(assistant_msg)

        # ── Execute each tool and collect results ─────────────────────────────
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

            # Track success for BOTH the validator and terminal-tool detection.
            if _tool_succeeded(result_json):
                successful_tool_calls.append(name)
                if name in TERMINAL_TOOLS:
                    terminal_tool = name

            tool_result_msg = {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_json,
            }

            # Append to both working list and history
            messages.append(tool_result_msg)
            history.append(tool_result_msg)

        log.info("tool_round", extra={
            "session": session_id,
            "iteration": iteration,
            "tools": [tc.function.name for tc in msg.tool_calls],
        })

    else:
        # Loop exhausted without a plain reply — force a fallback.
        final_text = (
            "Sorry, I got a bit tangled up there. Could you tell me again what "
            "you'd like to order?"
        )
        history.append({"role": "assistant", "content": final_text})
        log.warning("tool_loop_exhausted", extra={"session": session_id})

    # ── Persist full history for this turn ────────────────────────────────────
    # history now contains:
    #   ...previous turns...
    #   user message
    #   assistant (tool_calls) × N rounds
    #   tool results × N rounds
    #   assistant (final text)
    log.info("turn_complete", extra={"session": session_id, "messages": history})
    await save_history(session_id, history)

    return {
        "reply": final_text,
        "terminal_tool": terminal_tool,
        "tool_calls": called_tools,
    }