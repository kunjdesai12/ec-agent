"""EasyCater LLM API — FastAPI entrypoint.

Endpoints:
  POST /auth/send-otp               — send OTP to user's phone (login)
  POST /auth/verify-otp             — verify OTP and get JWT token
  POST /v1/chat                     — SSE streaming chat
  POST /v1/chat/sync                — non-streaming variant (debug/tests)
  POST /v1/chat/voice               — voice chat (STT → chat pipeline)
  GET  /v1/health                   — liveness + Redis ping
  GET  /v1/session/{sid}/cart       — read the in-progress cart
  POST /v1/session/{sid}/checkout/complete — clear cart after order placed
  POST /v1/session/{sid}/reset      — clear conversation history + cart

Brain: the tool-calling orchestrator (agent.app.orchestrator.run_turn). It owns
message history (in-memory) and drives the LLM ↔ tool loop. Order state lives in
the cart module (agent.app.cart), not order_params. The model hands off to the
checkout screen by emitting CONFIRMED on its own line; we strip that here and
return the cart payload for the screen.
"""
from __future__ import annotations
import os
import sys
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import httpx
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agent.app.config import get_settings
from agent.app.graph.orchestrator import run_turn, clear_history
from agent.app import cart
from agent.app.guardrails_middleware import AkiGuardrails
from agent.app.logging_setup import configure_logging, get_logger
from agent.app.rag.retriever import get_retriever
from agent.app.state import get_redis

configure_logging()
log = get_logger(__name__)

guardrails = AkiGuardrails()

STT_API_ENDPOINT = get_settings().stt_api_endpoint

# The model emits this on its own line to hand off to the checkout screen.
CONFIRMED_SENTINEL = "CONFIRMED"

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    # Warm up retriever (loads bge-m3 ~2GB into RAM) so the first
    # confirm_restaurant / confirm_item / search_food call doesn't block.
    log.info("startup_begin")
    get_retriever()
    # Initialize guardrails
    await guardrails.initialize()
    # Ping Redis
    r = await get_redis()
    await r.ping()
    log.info("startup_ready")
    yield
    log.info("shutdown")


app = FastAPI(title="EasyCater LLM API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────────────────────────────────
# Schemas
# ────────────────────────────────────────────────────────────────────────────

 
class SendOTPRequest(BaseModel):
    phone: str = Field(..., min_length=7, max_length=15)
    country_code: str = Field("+91")
 
class VerifyOTPRequest(BaseModel):
    phone: str = Field(..., min_length=7, max_length=15)
    otp: str = Field(..., min_length=4, max_length=8)
    country_code: str = Field("+91")

class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=4000)
    user_id: Optional[str] = None
    jwt_token: Optional[str] = None

class ChatSyncResponse(BaseModel):
    session_id: str
    text: str
    # New fields — the cart/checkout state for this turn. `text` is unchanged so
    # existing clients keep working; richer clients can read the rest.
    confirmed: bool = False
    checkout: Optional[dict] = None
    cart: Optional[dict] = None
    terminal_tool: Optional[str] = None

class VoiceJobResponse(BaseModel):
    session_id: str
    job_id: str

class VoiceJobResultRequest(BaseModel):
    session_id: str
    job_id: str


def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


# ────────────────────────────────────────────────────────────────────────────
# Turn helpers
# ────────────────────────────────────────────────────────────────────────────

def _split_confirmation(reply: str) -> tuple[str, bool]:
    """Strip the CONFIRMED sentinel from the reply → (clean_text, confirmed).

    The sentinel must be on its own line (per the prompt). We never show or
    speak it to the user.
    """
    lines = reply.splitlines()
    confirmed = any(line.strip() == CONFIRMED_SENTINEL for line in lines)
    if not confirmed:
        return reply, False
    cleaned = "\n".join(l for l in lines if l.strip() != CONFIRMED_SENTINEL).strip()
    return cleaned, True


def _finalize(session_id: str, result: dict) -> tuple[str, bool, Optional[dict]]:
    """Post-process a run_turn result: strip CONFIRMED, build checkout payload,
    and clear the cart on a placed order.

    Returns (clean_text, confirmed, checkout_payload).
    """
    text, confirmed = _split_confirmation(result["reply"])

    # On CONFIRMED, hand the cart to the checkout screen but keep it — the screen
    # needs the contents, and the user may return if they abandon checkout. The
    # cart is cleared by /checkout/complete once the order is actually placed.
    checkout = cart.summary(session_id) if confirmed else None

    # discard_current_order already cleared the cart in its handler. If a
    # place_order tool ran through the loop, clear here too. cancel_order is for
    # an already-PLACED order and must NOT touch the building cart.
    if result.get("terminal_tool") == "place_order":
        cart.clear(session_id)

    return text, confirmed, checkout


# ────────────────────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────────────────────

@app.post("/auth/send-otp")
async def send_otp(req: SendOTPRequest):
    """Send OTP to user's phone number.
 
    Forwards to the production backend. The caller only needs to provide
    phone and country_code — all other fields are hardcoded for the
    customer login flow.
    """
    payload = {
        "country_code":          req.country_code,
        "phone":                 req.phone,
        "type":                  "login",
        "user_type":             "user",
        "validate_company_user": "false",
        "formType":              "loginForm",
    }
 
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{get_settings().backend_base_url}/restaurants/send-otp-with-mobile",
                json=payload,
            )

            print("STATUS:", response.status_code)
            print("BODY:")
            print(response.text)

            try:
                data = response.json()
            except Exception:
                log.error("send_otp_bad_response", extra={
                    "status": response.status_code,
                    "body": response.text[:200],
                })
                raise HTTPException(
                    status_code=502,
                    detail=f"Auth service returned non-JSON response (status {response.status_code})"
                )
            
    except httpx.HTTPError as e:
        log.error("send_otp_failed", extra={"err": str(e)})
        raise HTTPException(status_code=502, detail="Could not reach auth service")
 
    if response.status_code not in (200, 201):
        raise HTTPException(
            status_code=response.status_code,
            detail=data.get("message", "Failed to send OTP"),
        )
 
    return {"ok": True, "message": data.get("message", "OTP sent successfully")}
 
 
@app.post("/auth/verify-otp")
async def verify_otp(req: VerifyOTPRequest):
    """Verify OTP and return JWT token.
 
    On success returns { ok, token, user_id, name } — the caller should
    pass token as jwt_token in all subsequent /v1/chat requests.
    """
    payload = {
        "country_code": req.country_code,
        "phone":        req.phone,
        "otp":          req.otp,
        "otp_type":     "login",
        "role_name":    "customer",
        "is_corporate": 0,
    }
 
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{get_settings().backend_base_url}/login/user-mobile",
                json=payload,
            )
            data = response.json()
    except httpx.HTTPError as e:
        log.error("verify_otp_failed", extra={"err": str(e)})
        raise HTTPException(status_code=502, detail="Could not reach auth service")
 
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=data.get("message", "OTP verification failed"),
        )
 
    token_data = data.get("data", {})
    token      = token_data.get("token")
    user       = token_data.get("users", {})
 
    if not token:
        log.error("verify_otp_no_token", extra={"phone": req.phone})
        raise HTTPException(status_code=502, detail="Auth service returned no token")
 
    log.info("user_logged_in", extra={"user_id": user.get("user_id"), "phone": req.phone})
 
    return {
        "ok":      True,
        "token":   token,
        "user_id": user.get("user_id"),
        "name":    user.get("name"),
    }


@app.get("/v1/health")
async def health():
    try:
        r = await get_redis()
        await r.ping()
        redis_ok = True
    except Exception as e:  # noqa: BLE001
        redis_ok = False
        log.error("redis_health_failed", extra={"err": str(e)})
    return {
        "ok": redis_ok,
        "redis": redis_ok,
        "model": get_settings().vllm_model,
        "guardrails": guardrails._initialized,
    }


@app.post("/v1/chat/sync", response_model=ChatSyncResponse)
async def chat_sync(req: ChatRequest):
    """Non-streaming chat — useful for tests and server-to-server calls."""
    # ── Guardrail input check ──────────────────────────────────
    guard = await guardrails.check_input(req.message, req.session_id)
    if guard["blocked"]:
        log.info(
            "guardrail_blocked",
            extra={"session_id": req.session_id, "violation": guard["violation"]},
        )
        return ChatSyncResponse(session_id=req.session_id, text=guard["response"])
    sys.stdout.write("\033[3J\033[H\033[2J")
    sys.stdout.flush()
    
    # ── Tool-calling loop (history + cart owned by orchestrator/cart) ──
    result = await run_turn(
        req.session_id, req.message, jwt_token=req.jwt_token or "",
    )
    text, confirmed, checkout = _finalize(req.session_id, result)
 
    
    return ChatSyncResponse(
        session_id=req.session_id,
        text=text,
        confirmed=confirmed,
        checkout=checkout,
        cart=cart.summary(req.session_id),
        terminal_tool=result.get("terminal_tool"),
    )


@app.post("/v1/chat")
async def chat_stream(req: ChatRequest):
    """SSE chat. Events: status, tool, token, checkout, done, error.

    NOTE: the tool-calling loop is not token-streaming, so the full reply is
    emitted as a single `token` event. Tool names are emitted after the turn
    completes. For true token-by-token streaming we'd need a streaming variant of
    run_turn — see notes.
    """
    async def event_gen():
        # ── Guardrail input check ──────────────────────────────
        guard = await guardrails.check_input(req.message, req.session_id)
        if guard["blocked"]:
            log.info(
                "guardrail_blocked",
                extra={"session_id": req.session_id, "violation": guard["violation"]},
            )
            yield {"event": "token", "data": json.dumps({"text": guard["response"]})}
            yield {"event": "done",  "data": json.dumps({"ok": True})}
            return

        try:
            yield {"event": "status", "data": json.dumps({"state": "thinking"})}

            result = await run_turn(
                req.session_id, req.message, jwt_token=req.jwt_token or "",
            )

            for name in result.get("tool_calls", []):
                yield {"event": "tool", "data": json.dumps({"name": name})}

            text, confirmed, checkout = _finalize(req.session_id, result)
            yield {"event": "token", "data": json.dumps({"text": text})}

            if confirmed:
                yield {"event": "checkout", "data": json.dumps({"checkout": checkout})}

            yield {"event": "done", "data": json.dumps({
                "ok": True,
                "confirmed": confirmed,
                "terminal_tool": result.get("terminal_tool"),
                "cart": cart.summary(req.session_id),
            })}

        except asyncio.CancelledError:
            log.info("client_disconnect", extra={"session_id": req.session_id})
            raise
        except Exception as e:  # noqa: BLE001
            log.error("chat_stream_failed", extra={"session_id": req.session_id, "err": str(e)})
            yield {"event": "error", "data": json.dumps({"error": "Aki had a problem handling that."})}

    return EventSourceResponse(event_gen())


@app.post("/v1/chat/voice", response_model=ChatSyncResponse)
async def chat_voice(
    session_id: str           = Form(...),
    audio:      UploadFile    = File(...),
    jwt_token:  Optional[str] = Form(None),
):
    """Voice chat — transcribes audio via STT, then runs the normal chat pipeline."""
    audio_bytes = await audio.read()
 
    async with httpx.AsyncClient() as client:
        stt_response = await client.post(
            STT_API_ENDPOINT,
            files={"file": (audio.filename, audio_bytes, audio.content_type)},
            timeout=100.0,
        )
        stt_response.raise_for_status()
 
    stt_data         = stt_response.json()
    result_stt       = stt_data.get("result", {})
    transcribed_text = result_stt.get("translated_text") or result_stt.get("text")
 
    if not transcribed_text:
        log.warning(
            "stt_empty_transcript",
            extra={"session_id": session_id, "job_id": stt_data.get("job_id")},
        )
        return ChatSyncResponse(
            session_id=session_id,
            text="Sorry, I couldn't understand the audio. Please try again.",
        )
 
    guard = await guardrails.check_input(transcribed_text, session_id)
    if guard["blocked"]:
        log.info(
            "guardrail_blocked",
            extra={"session_id": session_id, "violation": guard["violation"]},
        )
        return ChatSyncResponse(session_id=session_id, text=guard["response"])
 
    result = await run_turn(
        session_id, transcribed_text, jwt_token=jwt_token or "",
    )
    text, confirmed, checkout = _finalize(session_id, result)
 
    return ChatSyncResponse(
        session_id=session_id,
        text=text,
        confirmed=confirmed,
        checkout=checkout,
        cart=cart.summary(session_id),
        terminal_tool=result.get("terminal_tool"),
    )


@app.get("/v1/session/{session_id}/cart")
async def get_session_cart(session_id: str):
    """Current cart for a session — for the checkout screen to read."""
    return cart.summary(session_id)


@app.post("/v1/session/{session_id}/checkout/complete")
async def checkout_complete(session_id: str):
    """Called by the checkout screen AFTER the order is successfully placed.

    Clears the building cart so the next conversation starts clean.
    """
    cart.clear(session_id)
    return {"ok": True, "session_id": session_id}


@app.post("/v1/session/{session_id}/reset")
async def reset_session(session_id: str):
    """Clear conversation history and the in-progress cart (logout / start over)."""
    clear_history(session_id)
    cart.clear(session_id)
    return {"ok": True, "session_id": session_id}
