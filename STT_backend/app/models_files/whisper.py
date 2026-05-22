import time
import torch
import numpy as np
import librosa
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from app.config import MODEL_PATH, LANGUAGE, TASK, SAMPLING_RATE, STT_BACKEND

processor = None
model = None
DEVICE = None
DTYPE = None


def _resolve_local_model_path(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Whisper model path not found: {path}")
    return str(path.resolve())


def load_whisper_model():
    global processor, model, DEVICE, DTYPE
    print(f" Loading local Whisper model: {MODEL_PATH}")

    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

    print(f" Device: {DEVICE.upper()}")

    t0 = time.time()
    local_model_path = _resolve_local_model_path(MODEL_PATH)
    processor = WhisperProcessor.from_pretrained(
        local_model_path,
        local_files_only=True,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        local_model_path,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(DEVICE)
    model.eval()
    print(f" Model loaded in {time.time() - t0:.1f}s")


def _load_audio(audio_path: str):
    audio, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
    if len(audio) == 0:
        raise RuntimeError("Audio is empty")
    return audio.astype(np.float32)


def transcribe_local(audio_path: str) -> dict:
    audio = _load_audio(audio_path)
    duration = len(audio) / SAMPLING_RATE

    inputs = processor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt").input_features.to(DEVICE, dtype=DTYPE)

    forced_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK) if LANGUAGE else None

    t_start = time.perf_counter()
    with torch.no_grad():
        predicted_ids = model.generate(inputs, forced_decoder_ids=forced_ids, num_beams=1, use_cache=True)
    inference_sec = time.perf_counter() - t_start

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    rtf = round(inference_sec / duration, 4) if duration > 0 else 0

    return {
        "backend": "local",
        "text": text,
        "duration_sec": round(duration, 2),
        "inference_sec": round(inference_sec, 3),
        "rtf": rtf,
        "realtime": rtf < 1.0,
        "device": DEVICE,
    }