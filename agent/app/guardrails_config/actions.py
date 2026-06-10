# config/actions.py
# Custom NeMo Guardrails actions
# These bridge guardrail checks with your existing LangGraph pipeline

from typing import Optional
from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult


COMPETITORS = [
    "swiggy", "zomato", "blinkit", "dunzo", "zepto",
    "magicpin", "uber eats", "foodpanda"
]

OFF_TOPIC_KEYWORDS = [
    "weather", "cricket", "ipl", "stock", "share price",
    "politics", "news", "joke", "poem", "homework",
    "homework", "study", "exam", "movie", "song"
]


# ─── Input Checks ─────────────────────────────────────────────────────────────

@action(name="check_input_safety")
async def check_input_safety(context: Optional[dict] = None) -> ActionResult:
    """
    Fast regex-based check before hitting the LLM self-check.
    Catches obvious violations without LLM call = zero latency.
    """
    user_message = context.get("user_message", "").lower()

    # Check for competitors
    for competitor in COMPETITORS:
        if competitor in user_message:
            return ActionResult(
                return_value=False,
                context_updates={"violation_reason": f"competitor_mention:{competitor}"}
            )

    # Check for price manipulation keywords
    manipulation_keywords = [
        "free order", "zero price", "100% discount",
        "override price", "hack price", "set price to 1"
    ]
    for keyword in manipulation_keywords:
        if keyword in user_message:
            return ActionResult(
                return_value=False,
                context_updates={"violation_reason": "order_manipulation"}
            )

    return ActionResult(return_value=True)


@action(name="check_response_on_topic")
async def check_response_on_topic(context: Optional[dict] = None) -> ActionResult:
    """
    Check if bot response is on topic (food/ordering related).
    Fast keyword check — LLM self-check handles edge cases.
    """
    bot_response = context.get("bot_response", "").lower()

    # If response mentions off-topic subjects, flag it
    for keyword in OFF_TOPIC_KEYWORDS:
        if keyword in bot_response:
            return ActionResult(return_value=False)

    return ActionResult(return_value=True)


@action(name="check_competitor_in_response")
async def check_competitor_in_response(context: Optional[dict] = None) -> ActionResult:
    """Check if bot response mentions any competitor."""
    bot_response = context.get("bot_response", "").lower()

    for competitor in COMPETITORS:
        if competitor in bot_response:
            return ActionResult(
                return_value=True,  # True = competitor found = block
                context_updates={"violation_reason": f"competitor_in_response:{competitor}"}
            )

    return ActionResult(return_value=False)


# ─── Order Validation ─────────────────────────────────────────────────────────

@action(name="validate_order_before_placement")
async def validate_order_before_placement(context: Optional[dict] = None) -> ActionResult:
    """
    Called before place_order tool executes.
    Validates that all required fields are present and sane.
    """
    order_data = context.get("pending_order", {})

    errors = []

    if not order_data.get("restaurant_id"):
        errors.append("missing restaurant")

    if not order_data.get("items"):
        errors.append("no items selected")

    if not order_data.get("delivery_address"):
        errors.append("missing delivery address")

    if not order_data.get("phone_number"):
        errors.append("missing phone number")

    # Sanity check: max 10 items, max ₹5000 total
    items = order_data.get("items", [])
    if len(items) > 10:
        errors.append("too many items (max 10)")

    total = order_data.get("estimated_total", 0)
    if total > 5000:
        errors.append("order total exceeds ₹5000 limit")

    if errors:
        return ActionResult(
            return_value=False,
            context_updates={"validation_errors": errors}
        )

    return ActionResult(return_value=True)


@action(name="check_confirmation_given")
async def check_confirmation_given(context: Optional[dict] = None) -> ActionResult:
    """
    Ensure user explicitly confirmed before place_order is called.
    Prevents accidental orders.
    """
    conversation_state = context.get("conversation_state", {})
    confirmed = conversation_state.get("order_confirmed", False)

    return ActionResult(return_value=confirmed)


# ─── LangGraph Bridge ─────────────────────────────────────────────────────────

@action(name="process_with_langgraph")
async def process_with_langgraph(context: Optional[dict] = None) -> ActionResult:
    """
    Passes the validated user message to your existing LangGraph Aki pipeline.
    This is the main bridge between NeMo and your agent.
    """
    import httpx

    user_message = context.get("user_message", "")
    session_id   = context.get("session_id", "default")
    history      = context.get("conversation_history", [])

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8000/agent/chat",  # your LangGraph FastAPI endpoint
                json={
                    "message":  user_message,
                    "session_id": session_id,
                    "history":  history,
                }
            )
            data = response.json()
            return ActionResult(
                return_value=data.get("response", ""),
                context_updates={
                    "last_tool_called": data.get("tool_called"),
                    "order_id":         data.get("order_id"),
                    "conversation_state": data.get("state", {}),
                }
            )
    except Exception as e:
        return ActionResult(
            return_value="I'm having trouble connecting right now. Please try again.",
            context_updates={"error": str(e)}
        )


# ─── Logging ──────────────────────────────────────────────────────────────────

@action(name="log_guardrail_violation")
async def log_guardrail_violation(context: Optional[dict] = None) -> ActionResult:
    """
    Log all guardrail violations for monitoring.
    Hook this into your existing Winston/OpenTelemetry logging.
    """
    import json
    from datetime import datetime

    violation = {
        "timestamp":    datetime.utcnow().isoformat(),
        "session_id":   context.get("session_id"),
        "user_message": context.get("user_message"),
        "reason":       context.get("violation_reason", "unknown"),
        "bot_response": context.get("bot_response"),
    }

    # Write to log file — replace with your logging stack
    with open("/data/ec-agent/logs/guardrail_violations.jsonl", "a") as f:
        f.write(json.dumps(violation) + "\n")

    return ActionResult(return_value=True)