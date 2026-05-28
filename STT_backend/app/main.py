import asyncio
from fastapi import FastAPI
from app.config import STT_BACKEND, ENABLE_TRANSLATION
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
    asyncio.create_task(_worker())

    if STT_BACKEND:
        from app.models_files.whisper import load_whisper_model
        load_whisper_model()
    else:
        from app.models_files.sarvam import init_sarvam_client
        init_sarvam_client()

    if ENABLE_TRANSLATION:
        from app.models_files.translator import init_translator_model
        init_translator_model()


@app.get("/health")
def health():
    return {
        "status":  "ok",
        "backend": "local_whisper" if STT_BACKEND else "sarvam",
        "queue":   "python_asyncio",
        "translation": "enabled" if ENABLE_TRANSLATION else "disabled",
    }