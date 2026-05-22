from fastapi import FastAPI
from app.config import STT_BACKEND, HOST, PORT
from app.models.whisper import load_whisper_model
from app.api.transcribe import router as transcribe_router

app = FastAPI(
    title="Speech-to-Text API",
    description="Switchable STT backend (Sarvam / Local Whisper)",
    version="1.0.0",
)

app.include_router(transcribe_router)

if STT_BACKEND:
    load_whisper_model()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "local" if STT_BACKEND else "sarvam",
        "device": "cpu/mps" if STT_BACKEND else "cloud",
    }