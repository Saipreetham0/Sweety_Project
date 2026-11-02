import os
import re
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter(prefix="/upload", tags=["upload"])

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(_BACKEND_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_BYTES = 10 * 1024 * 1024  # 10 MB


def sanitize_filename(filename: str) -> str:
    """Sanitizes filename and prevents directory traversal attacks."""
    base = os.path.basename(filename or "resume.txt")
    clean = re.sub(r"[^\w\s\.-]", "_", base).strip()
    return clean or "resume.txt"


@router.post("/")
async def upload_resume(file: UploadFile = File(...)):
    raw_name = file.filename or "resume.txt"
    ext = os.path.splitext(raw_name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed types: PDF, DOCX, TXT."
        )

    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB).")

    # Unique prefix so concurrent uploads with the same name don't overwrite each other.
    # ponytail: uploads are never deleted; add a cleanup job if disk usage matters.
    safe_name = f"{uuid.uuid4().hex[:8]}_{sanitize_filename(raw_name)}"
    with open(os.path.join(UPLOAD_DIR, safe_name), "wb") as buffer:
        buffer.write(data)

    return {"filename": safe_name, "status": "uploaded"}
