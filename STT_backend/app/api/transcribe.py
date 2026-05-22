import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.utils.audio import validate_audio_file, save_temp_file
from app.models.whisper import transcribe_local
from app.models.sarvam import transcribe_sarvam
from app.models.translator import translate_to_english
from app.config import STT_BACKEND, ENABLE_TRANSLATION

router = APIRouter(tags=["stt"])


@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    validate_audio_file(file.filename)
    tmp_path = save_temp_file(file)

    try:
        if not STT_BACKEND:
            result = transcribe_sarvam(tmp_path)
        else:
            result = transcribe_local(tmp_path)

        if ENABLE_TRANSLATION:
            result["translated_text"] = translate_to_english(result["text"])

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)