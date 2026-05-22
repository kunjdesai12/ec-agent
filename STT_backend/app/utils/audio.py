import os
import tempfile
from fastapi import HTTPException

from app.config import SUPPORTED_EXT

def validate_audio_file(filename: str):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Use: {', '.join(SUPPORTED_EXT)}"
        )

def save_temp_file(upload_file) -> str:
    ext = os.path.splitext(upload_file.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(upload_file.file.read())
        return tmp.name