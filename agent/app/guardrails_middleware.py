# guardrails_middleware.py
#
# Place at: /data/ec-agent/agent/app/guardrails_middleware.py
#
# Wraps the Aki LangGraph pipeline with NeMo Guardrails input checks.
# Only input rails are used here — LangGraph handles the actual response,
# so no output rails are needed at this layer (keeps latency minimal).

import os
import logging
from typing import Optional

from nemoguardrails import RailsConfig, LLMRails

log = logging.getLogger(__name__)

# Points to the guardrails_config/ subfolder next to this file
GUARDRAILS_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "guardrails_config")


class AkiGuardrails:
    """
    Lightweight guardrail gate that sits in front of run_turn / stream_turn.

    Only runs INPUT rails — jailbreak, off-topic, competitor, price manipulation.
    LangGraph handles the actual Aki response unchanged.

    Guardrail layers (in order, fast to slow):
      1. Regex check      — zero latency, catches obvious violations
      2. Semantic rail    — NeMo embedding similarity, ~10-20ms
      3. LLM self-check   — Qwen via vLLM, ~200-400ms, only if needed
    """

    def __init__(self):
        self.rails: Optional[LLMRails] = None
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            return

        log.info("guardrails_init_begin")

        try:
            config = RailsConfig.from_path(GUARDRAILS_CONFIG_PATH)
            self.rails = LLMRails(config)

            # Register custom actions from guardrails_config/actions.py
            from agent.app.guardrails_config.actions import (
                check_input_safety,
                check_response_on_topic,
                check_competitor_in_response,
                validate_order_before_placement,
                check_confirmation_given,
                log_guardrail_violation,
            )
            self.rails.register_action(check_input_safety)
            self.rails.register_action(check_response_on_topic)
            self.rails.register_action(check_competitor_in_response)
            self.rails.register_action(validate_order_before_placement)
            self.rails.register_action(check_confirmation_given)
            self.rails.register_action(log_guardrail_violation)

            self._initialized = True
            log.info("guardrails_init_ready")

        except Exception as e:
            # Guardrails failing to load should NOT crash the whole app.
            # Log the error and continue — Aki will run without guardrails.
            log.error("guardrails_init_failed", extra={"err": str(e)})
            self._initialized = False

    async def check_input(self, user_message: str, session_id: str) -> dict:
        """
        Run input rails only against the user message.
        Returns immediately if a rail fires — LangGraph is never called.

        Returns:
            {
                "blocked":   bool,
                "response":  str,        # message to return to user if blocked
                "violation": str | None, # which rule fired
            }
        """
        # If guardrails failed to initialize, pass everything through
        if not self._initialized:
            return {"blocked": False, "response": "", "violation": None}

        context = {
            "session_id":   session_id,
            "user_message": user_message,
        }

        # ── Layer 1: Fast regex check (zero LLM calls) ─────────────────────
        from agent.app.guardrails_config.actions import check_input_safety
        safety_result = await check_input_safety(context=context)

        if not safety_result.return_value:
            violation = safety_result.context_updates.get("violation_reason", "input_unsafe")
            log.info("guardrail_regex_block", extra={
                "session_id": session_id,
                "violation":  violation,
            })
            await self._log_violation(session_id, user_message, violation)
            return {
                "blocked":   True,
                "response":  self._blocked_response(violation),
                "violation": violation,
            }

        # ── Layer 2: NeMo semantic + LLM self-check ────────────────────────
        try:
            result = await self.rails.generate_async(
                messages=[{"role": "user", "content": user_message}],
                options={
                    "rails": ["input"],   # input rails only, no LLM generation
                    "output_vars": True,
                },
            )

            # If NeMo generated a canned response (rail fired), it means blocked
            response_text = ""
            if result.response:
                response_text = result.response[-1].get("content", "")

            # A rail fired if NeMo returned a canned bot response
            # (not an empty string and not a passthrough)
            blocked = bool(response_text) and not self._is_passthrough(response_text)
            violation = result.context.get("violation_reason") if blocked else None

            if blocked:
                log.info("guardrail_semantic_block", extra={
                    "session_id": session_id,
                    "violation":  violation,
                })
                await self._log_violation(session_id, user_message, violation or "semantic_rail")

            return {
                "blocked":   blocked,
                "response":  response_text if blocked else "",
                "violation": violation,
            }

        except Exception as e:
            # If NeMo check fails, log and pass through — don't break the chat
            log.error("guardrail_check_error", extra={"err": str(e), "session_id": session_id})
            return {"blocked": False, "response": "", "violation": None}

    def _blocked_response(self, violation: str) -> str:
        """Return the right canned message based on which rule fired."""
        if "competitor" in violation:
            return (
                "I can only help with orders on EasyCater. "
                "We have Rajwadu, Pizza Hub, and Biryani Blues available. "
                "Shall I show you their menus?"
            )
        if "manipulation" in violation or "price" in violation:
            return (
                "I'm not able to modify prices or apply unauthorized discounts. "
                "All prices are as listed on our menu. Would you like to see what's available?"
            )
        if "jailbreak" in violation:
            return (
                "I'm Aki, EasyCater's food ordering assistant. "
                "What would you like to eat today?"
            )
        # Default off-topic / unsafe
        return (
            "I can only help with food orders on EasyCater. "
            "Would you like to see our menu?"
        )

    def _is_passthrough(self, response: str) -> bool:
        """
        NeMo sometimes returns the original message unchanged when no rail fires.
        Detect this so we don't treat it as a block.
        """
        passthrough_signals = [
            "i want to order",
            "show me the menu",
            "what's available",
        ]
        r = response.lower()
        return any(s in r for s in passthrough_signals)

    async def _log_violation(self, session_id: str, message: str, violation: str):
        """Fire-and-forget violation log."""
        try:
            import json
            from datetime import datetime

            log_dir = "/data/ec-agent/logs"
            os.makedirs(log_dir, exist_ok=True)

            record = {
                "timestamp":    datetime.utcnow().isoformat(),
                "session_id":   session_id,
                "user_message": message,
                "violation":    violation,
            }
            with open(f"{log_dir}/guardrail_violations.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass  # logging failure should never affect the chat