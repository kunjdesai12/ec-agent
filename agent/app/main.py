"""EasyCater LLM API — FastAPI entrypoint.

Endpoints:
  POST /v1/auth/send-otp            — send OTP to user's phone (login)
  POST /v1/auth/verify-otp          — verify OTP and get JWT token
  POST /v1/chat                     — SSE streaming chat
  POST /v1/chat/sync                — non-streaming variant (debug/tests)
  POST /v1/chat/voice               — voice chat (STT → chat pipeline)
  GET  /v1/health                   — liveness + Redis ping
  POST /v1/session/{sid}/reset      — clear conversation history
"""
from __future__ import annotations
import os
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
from agent.app.graph import run_turn
from agent.app.graph.streaming import stream_turn
from agent.app.guardrails_middleware import AkiGuardrails
from agent.app.logging_setup import configure_logging, get_logger
from agent.app.rag import get_retriever
from agent.app.state import get_redis, load_history, reset_history, save_history
from agent.app.state.history import load_order_params, save_order_params

configure_logging()
log = get_logger(__name__)

guardrails = AkiGuardrails()

STT_API_ENDPOINT = get_settings().stt_api_endpoint

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    # Warm up retriever (loads bge-m3 ~2GB into RAM)
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

class VoiceJobResponse(BaseModel):
    session_id: str
    job_id: str

class VoiceJobResultRequest(BaseModel):
    session_id: str
    job_id: str


def clear_console():
    os.system("cls" if os.name == "nt" else "clear")

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
    clear_console()
    # ── Guardrail input check ──────────────────────────────────
    guard = await guardrails.check_input(req.message, req.session_id)
    if guard["blocked"]:
        log.info(
            "guardrail_blocked",
            extra={"session_id": req.session_id, "violation": guard["violation"]},
        )
        return ChatSyncResponse(session_id=req.session_id, text=guard["response"])

    # ── Normal pipeline ────────────────────────────────────────
    history = await load_history(req.session_id)
    order_params = await load_order_params(req.session_id)

    text, new_history, updated_order_params = await run_turn(
        req.session_id, req.message, history, order_params, jwt_token=req.jwt_token or "",
    )

    await save_history(
        req.session_id,
        _dedup_history(
            history
            + [{"role": "user", "content": req.message}]
            + _without_user_dup(new_history, req.message)
        ),
    )

    await save_order_params(req.session_id, updated_order_params)

    return ChatSyncResponse(session_id=req.session_id, text=text)


@app.post("/v1/chat")
async def chat_stream(req: ChatRequest):
    """SSE streaming chat. Events:
       status, tool, token, done, error — see app/graph/streaming.py
    """
    history = await load_history(req.session_id)
    order_params = await load_order_params(req.session_id)

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

        # ── Normal pipeline ────────────────────────────────────
        persistable: list[dict] | None = None
        updated_params: dict = order_params
        try:
            async for event in stream_turn(
                req.session_id, req.message, history, order_params, jwt_token=req.jwt_token or "",
            ):
                if event["type"] == "done":
                    persistable = event["messages"]
                    updated_params = event.get("order_params", {})
                    yield {"event": "done", "data": json.dumps({"ok": True})}
                else:
                    yield {"event": event["type"], "data": json.dumps(event)}

            # Persist after the stream finishes successfully
            if persistable is not None:
                new_hist = history + persistable
                await save_history(req.session_id, new_hist)
                await save_order_params(req.session_id, updated_params)

        except asyncio.CancelledError:
            log.info("client_disconnect", extra={"session_id": req.session_id})
            raise

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
    result           = stt_data.get("result", {})
    transcribed_text = result.get("translated_text") or result.get("text")
 
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
 
    history      = await load_history(session_id)
    order_params = await load_order_params(session_id)
 
    text, new_history, updated_order_params = await run_turn(
        session_id,
        transcribed_text,
        history,
        order_params,
        jwt_token=jwt_token or "",
    )
 
    await save_history(
        session_id,
        _dedup_history(
            history
            + [{"role": "user", "content": transcribed_text}]
            + _without_user_dup(new_history, transcribed_text)
        ),
    )

    await save_order_params(session_id, updated_order_params)
 
    return ChatSyncResponse(session_id=session_id, text=text)


@app.post("/v1/session/{session_id}/reset")
async def reset_session(session_id: str):
    await reset_history(session_id)
    await save_order_params(session_id, {})  # clear order params too
    return {"ok": True, "session_id": session_id}


def _without_user_dup(new_msgs: list[dict], user_msg: str) -> list[dict]:
    """Strip a leading user message from new_msgs if it duplicates user_msg.

    `run_turn` returns messages that include the user message we just added,
    because we built initial_messages with it. We avoid double-storing.
    """
    if new_msgs and new_msgs[0].get("role") == "user" and new_msgs[0].get("content") == user_msg:
        return new_msgs[1:]
    return new_msgs

def _dedup_history(messages: list[dict]) -> list[dict]:
    """Remove duplicate message blocks from history.
    
    Deduplicates both simple consecutive duplicates and repeated
    tool call + tool result pairs that accumulate across turns.
    """
    if not messages:
        return messages

    # Build a fingerprint for each message
    def _fingerprint(m: dict) -> str:
        role = m.get("role", "")
        content = m.get("content") or ""
        # For assistant messages with tool_calls, include the tool call id
        tool_calls = m.get("tool_calls")
        if tool_calls:
            tc_sig = ",".join(tc["function"]["name"] for tc in tool_calls)
            return f"{role}:tool_calls:{tc_sig}"
        tool_call_id = m.get("tool_call_id", "")
        if tool_call_id:
            return f"{role}:tool_result:{tool_call_id}:{content[:100]}"
        return f"{role}:{content[:200]}"

    seen: set[str] = set()
    deduped: list[dict] = []
    for msg in messages:
        fp = _fingerprint(msg)
        if fp not in seen:
            seen.add(fp)
            deduped.append(msg)
    return deduped