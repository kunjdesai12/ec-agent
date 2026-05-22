import time
from sarvamai import SarvamAI
from app.config import SARVAM_API_KEY, STT_BACKEND

sarvam_client = None

if not STT_BACKEND:
    print(" Initializing Sarvam client...")
    sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    print(" Sarvam client ready")


def transcribe_sarvam(audio_path: str) -> dict:
    t_start = time.perf_counter()
    with open(audio_path, "rb") as f:
        response = sarvam_client.speech_to_text.transcribe(
            file=f, model="saaras:v3", mode="transcribe"
        )
    inference_sec = time.perf_counter() - t_start

    if hasattr(response, "dict"):
        data = response.dict()
    elif hasattr(response, "model_dump"):
        data = response.model_dump()
    else:
        data = dict(response)

    return {
        "backend": "sarvam",
        "text": data.get("transcript", ""),
        "raw_response": data,
        "inference_sec": round(inference_sec, 3),
    }