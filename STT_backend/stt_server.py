import os
import time
import tempfile

import numpy as np
import librosa
import torch
import uvicorn

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

from sarvamai import SarvamAI
from llama_cpp import Llama

load_dotenv()

_stt_backend_raw = os.getenv("STT_BACKEND", "true").strip().lower()
STT_BACKEND = _stt_backend_raw not in {"false", "0", "no", "sarvam"}

HOST = "0.0.0.0"
PORT = 8000

SUPPORTED_EXT = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
    ".webm",
}

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

SARVAM_MODEL = "saaras:v3"
SARVAM_MODE  = "transcribe"

MODEL_PATH = "./models/STT_whisper_Model"

LANGUAGE = None
TASK = "transcribe"

SAMPLING_RATE = 16_000

_translate_raw = os.getenv("ENABLE_TRANSLATION", "false").strip().lower()

ENABLE_TRANSLATION = _translate_raw in {
    "true",
    "1",
    "yes",
}

QWEN_MODEL_PATH = "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

app = FastAPI(
    title="Speech-to-Text API",
    description="Switchable STT backend (Sarvam / Local Whisper)",
    version="1.0.0",
)

sarvam_client = None

if not STT_BACKEND:

    print(" Initializing Sarvam client...")

    sarvam_client = SarvamAI(
        api_subscription_key=SARVAM_API_KEY,
    )

    print(" Sarvam client ready")

processor = None
model = None
DEVICE = None
DTYPE = None
translator_model = None

if STT_BACKEND:

    print(f" Loading local Whisper model: {MODEL_PATH}")

    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

    if DEVICE == "cuda":
        DTYPE = torch.float16
    else:
        DTYPE = torch.float32

    print(f" Device: {DEVICE.upper()}")

    t0 = time.time()

    processor = WhisperProcessor.from_pretrained(MODEL_PATH)

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)

    model.eval()

    print(f" Model loaded in {time.time() - t0:.1f}s")


if ENABLE_TRANSLATION:

    print(f" Loading translation model: {QWEN_MODEL_PATH}")

    translator_model = Llama(
        model_path=QWEN_MODEL_PATH,
        n_ctx=2048,
        n_threads=max(os.cpu_count() // 2, 1),
        verbose=False,
    )

    print(" Translation model ready")



@app.get("/health")
def health():

    return {
        "status": "ok",
        "backend": STT_BACKEND,
        "device": DEVICE if DEVICE else "cloud",
    }

def _load_audio(audio_path: str):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLING_RATE,
        mono=True,
    )

    if audio is None or len(audio) == 0:
        raise RuntimeError("Audio is empty")

    audio = audio.astype(np.float32)

    return audio, sr

def transcribe_sarvam(audio_path: str) -> dict:

    t_start = time.perf_counter()

    with open(audio_path, "rb") as audio_file:

        response = sarvam_client.speech_to_text.transcribe(
            file=audio_file,
            model=SARVAM_MODEL,
            mode=SARVAM_MODE,
        )

    inference_sec = time.perf_counter() - t_start

    # Safe conversion
    if hasattr(response, "dict"):
        response_data = response.dict()

    elif hasattr(response, "model_dump"):
        response_data = response.model_dump()

    else:
        response_data = dict(response)

    return {
        "backend": "sarvam",
        "text": response_data.get("transcript", ""),
        "raw_response": response_data,
        "inference_sec": round(inference_sec, 3),
    }

def transcribe_local(audio_path: str) -> dict:

    audio, sr = _load_audio(audio_path)

    duration = len(audio) / SAMPLING_RATE

    inputs = processor(
        audio,
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
    ).input_features.to(DEVICE, dtype=DTYPE)

    forced_ids = (
        processor.get_decoder_prompt_ids(
            language=LANGUAGE,
            task=TASK,
        )
        if LANGUAGE
        else None
    )

    t_start = time.perf_counter()

    with torch.no_grad():

        predicted_ids = model.generate(
            inputs,
            forced_decoder_ids=forced_ids,
            num_beams=1,
            use_cache=True,
        )

    inference_sec = time.perf_counter() - t_start

    text = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
    )[0].strip()

    rtf = (
        round(inference_sec / duration, 4)
        if duration > 0
        else 0
    )

    return {
        "backend": "local",
        "text": text,
        "duration_sec": round(duration, 2),
        "inference_sec": round(inference_sec, 3),
        "rtf": rtf,
        "realtime": rtf < 1.0,
        "device": DEVICE,
    }

def translate_to_english(text: str) -> str:

    if not text or not text.strip():
        return text

    system_prompt = """
    You are an expert multilingual translator for Indian languages.

    Translate the input text into fluent and natural English.

    Rules:
    - Preserve the original meaning accurately.
    - Understand cultural and contextual meaning correctly.
    - Food names, cuisine names, people groups, and places should NOT be mistranslated literally.
    - Do not hallucinate or invent meanings.
    - Keep named entities and cuisine references accurate.
    - Return ONLY the English translation.
    """

    prompt = f"""
    <|system|>
    {system_prompt}

    <|user|>
    {text}

    <|assistant|>
    """

    output = translator_model(
        prompt,
        max_tokens=128,
        temperature=0.5,
        top_p=0.9,
        stop=["<|user|>", "<|system|>"],
    )

    translated = output["choices"][0]["text"].strip()

    return translated

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):

    # Validate extension
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in SUPPORTED_EXT:

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. "
                   f"Use: {', '.join(SUPPORTED_EXT)}"
        )

    # Save temp file
    with tempfile.NamedTemporaryFile(
        suffix=ext,
        delete=False,
    ) as tmp:

        tmp.write(await file.read())

        tmp_path = tmp.name

    try:

        if not STT_BACKEND:

            result = transcribe_sarvam(tmp_path)

        elif STT_BACKEND:

            result = transcribe_local(tmp_path)

        else:

            raise RuntimeError(
                f"Invalid STT_BACKEND: {STT_BACKEND}"
            )


        if ENABLE_TRANSLATION:

            translated_text = translate_to_english(
                result["text"]
            )

            result["translated_text"] = translated_text

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return JSONResponse(content=result)

if __name__ == "__main__":

    uvicorn.run(
        "stt_server:app",
        host=HOST,
        port=PORT,
        reload=False,
    )