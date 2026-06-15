# guardrails_middleware.py
#
# Place at: /data/ec-agent/agent/app/guardrails_middleware.py
#
# Wraps the Aki LangGraph pipeline with input-side guardrail checks.
# LangGraph handles the actual response, so only INPUT gating happens here.
#
# Guardrail layers (in order, fast to slow):
#   1. Regex check       — zero latency, catches obvious violations
#   2. Semantic classify — bge-m3 cosine over intent classes, no LLM call.
#                          Scores the input against BOTH violation and benign
#                          classes and blocks only when a violation wins above
#                          a similarity floor. Replaces NeMo's dialog-intent
#                          input rails (which force-matched every input to the
#                          nearest violation, with no benign class / no floor).

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

    Runs INPUT checks only — jailbreak, off-topic, competitor, price
    manipulation, abuse. LangGraph handles the actual Aki response unchanged.
    """

    def __init__(self):
        self.rails: Optional[LLMRails] = None
        self.semantic = None
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

            # Semantic input classifier — reuse the retriever's bge-m3 instance
            # (do NOT load a second copy; get_retriever() returns the singleton
            # that main.py already warms up at startup).
            from agent.app.guardrails_config.semantic_rail import SemanticRail
            from agent.app.rag.retriever import get_retriever
            self.semantic = SemanticRail(get_retriever().model)

            self._initialized = True
            log.info("guardrails_init_ready")

        except Exception as e:
            # Guardrails failing to load should NOT crash the whole app.
            # Log the error and continue — Aki will run without guardrails.
            log.error("guardrails_init_failed", extra={"err": str(e)})
            self._initialized = False

    async def check_input(self, user_message: str, session_id: str) -> dict:
        """
        Run input checks against the user message.
        Returns immediately if a check fires — LangGraph is never called.

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

        # ── Layer 2: semantic classifier (bge-m3 cosine, no LLM) ───────────
        try:
            verdict = self.semantic.check(user_message)
            if verdict["blocked"]:
                violation = verdict["violation"]
                log.info("guardrail_semantic_block", extra={
                    "session_id": session_id,
                    "violation":  violation,
                    "score":      round(verdict["score"], 3),
                })
                await self._log_violation(session_id, user_message, violation)
                return {
                    "blocked":   True,
                    "response":  self._blocked_response(violation),
                    "violation": violation,
                }

            return {"blocked": False, "response": "", "violation": None}

        except Exception as e:
            # If the semantic check fails, log and pass through — don't break chat
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
        if "abusive" in violation:
            return (
                "I'm here to help with your food order. "
                "What can I get started for you?"
            )
        # Default off-topic / unsafe
        return (
            "I can only help with food orders on EasyCater. "
            "Would you like to see our menu?"
        )

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