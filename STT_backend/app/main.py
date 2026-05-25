import asyncio
from fastapi import FastAPI
from app.config import STT_BACKEND, HOST, PORT
from app.models_files.whisper import load_whisper_model
from app.api.transcribe import router as transcribe_router
from app.api.transcribe import _worker

app = FastAPI(
    title="Speech-to-Text API",
    description="Switchable STT backend with async queue",
    version="2.0.0",
)

app.include_router(transcribe_router)


@app.on_event("startup")
async def startup():
    # Start background worker when app starts
    asyncio.create_task(_worker())
    if STT_BACKEND:
        load_whisper_model()


@app.get("/health")
def health():
    return {
        "status":  "ok",
        "backend": "local" if STT_BACKEND else "sarvam",
        "queue":   "python_asyncio",
        "device":  "cpu/mps" if STT_BACKEND else "cloud",
    }