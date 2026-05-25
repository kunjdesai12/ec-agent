import os
from pathlib import Path
from dotenv import load_dotenv
import torch

load_dotenv()   # Loads root .env

# Backend Settings
STT_BACKEND = os.getenv("STT_BACKEND", "true").strip().lower() not in {"false", "0", "no", "sarvam"}

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
ENABLE_TRANSLATION = os.getenv("ENABLE_TRANSLATION", "false").strip().lower() in {"true", "1", "yes"}

# Models are at project root level
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "STT_whisper_Model"
QWEN_MODEL_PATH = ROOT_DIR / "models" / "qwen-food-merged.Q4_K_M.gguf"

print(f"Model path: {MODEL_PATH.resolve()}")
print(f"Exists: {MODEL_PATH.exists()}")
# Server & Constants
HOST = "0.0.0.0"
PORT = 8000
SUPPORTED_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"}
LANGUAGE = None
TASK = "transcribe"
SAMPLING_RATE = 16_000