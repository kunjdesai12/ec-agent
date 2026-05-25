import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.utils.audio import validate_audio_file, save_temp_file
from app.config import STT_BACKEND, ENABLE_TRANSLATION
import uuid

router = APIRouter(tags=["stt"])

_jobs: dict = {}
_executor = ThreadPoolExecutor(max_workers=2)
_queue = asyncio.Queue()

# Decide at startup based on .env
if STT_BACKEND:
    from app.models_files.whisper import transcribe_local as transcribe_stt
else:
    from app.models_files.sarvam import transcribe_sarvam as transcribe_stt


async def _worker():
    while True:
        job_id, tmp_path = await _queue.get()
        try:
            _jobs[job_id]["status"] = "processing"
            loop = asyncio.get_event_loop()

            result = await loop.run_in_executor(_executor, transcribe_stt, tmp_path)

            if ENABLE_TRANSLATION:
                from app.models_files.translator import translate_to_english
                translated = await loop.run_in_executor(
                    _executor, translate_to_english, result["text"]
                )
                result["translated_text"] = translated

            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = result

        except Exception as e:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"]  = str(e)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            _queue.task_done()


@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    validate_audio_file(file.filename)
    tmp_path = save_temp_file(file)

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "result": None, "error": None}

    await _queue.put((job_id, tmp_path))

    return {
        "job_id":  job_id,
        "status":  "queued",
        "message": f"Poll /status/{job_id} for result"
    }


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]

    if job["status"] == "completed":
        return {"job_id": job_id, "status": "completed", "result": job["result"]}
    elif job["status"] == "failed":
        return {"job_id": job_id, "status": "failed", "error": job["error"]}
    else:
        return {"job_id": job_id, "status": job["status"]}