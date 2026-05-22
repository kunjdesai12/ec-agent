import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()   # Loads root .env

# Backend Settings
STT_BACKEND = os.getenv("STT_BACKEND", "true").strip().lower() not in {"false", "0", "no", "sarvam"}

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
ENABLE_TRANSLATION = os.getenv("ENABLE_TRANSLATION", "false").strip().lower() in {"true", "1", "yes"}

# Models are at project root level
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "STT_whisper_model"
QWEN_MODEL_PATH = ROOT_DIR / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"

# Server & Constants
HOST = "0.0.0.0"
PORT = 8000
SUPPORTED_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"}
LANGUAGE = None
TASK = "transcribe"
SAMPLING_RATE = 16_000