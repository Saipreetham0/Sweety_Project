"""
POST /analyze/{filename}

Runs the full 4-step pipeline:
  1. Text extraction (PDF, DOCX, TXT)
  2. Preprocessing (preserves structural layout for bullet/section detection)
  3. Weak supervision (heuristics from Dataset B)
  4. Hybrid RF + XGBoost ensemble prediction
"""

from __future__ import annotations

import os
from fastapi import APIRouter, HTTPException, Query

from services.extraction import TextExtractor
from services.preprocessing import TextPreprocessor
from services.weak_supervision import WeakSupervision
from services.model import ResumeModel

router = APIRouter(prefix="/analyze", tags=["analyze"])

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(_BACKEND_DIR, "uploads")
# Uploads first, then the bundled sample corpus (used by the frontend's Quick Test buttons)
_SEARCH_DIRS = [
    UPLOAD_DIR,
    os.path.join(_BACKEND_DIR, "data", "ai"),
    os.path.join(_BACKEND_DIR, "data", "human"),
    os.path.join(_BACKEND_DIR, "data", "template"),
]

# Singletons — load once at startup so model weights are not reloaded per request
_model = ResumeModel()
_ws = WeakSupervision()


@router.post("/{filename}")
def analyze_resume(filename: str, skip_perplexity: bool = Query(True)):
    # basename() strips any directory component, so lookups stay inside _SEARCH_DIRS
    safe_name = os.path.basename(filename)
    file_path = next(
        (p for p in (os.path.join(d, safe_name) for d in _SEARCH_DIRS) if os.path.isfile(p)),
        None,
    )
    if not safe_name or file_path is None:
        raise HTTPException(status_code=404, detail=f"File '{safe_name}' not found in uploads")

    # 1. Extraction
    try:
        text = TextExtractor.extract_from_file(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Extraction failed: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Extracted text is empty or unreadable.")

    # 2. Preprocessing
    clean_text = TextPreprocessor.clean_text(text)

    # 3. Weak Supervision
    heuristics = _ws.apply_heuristics(clean_text)

    # 4. Hybrid ML Prediction
    result = _model.predict(clean_text, heuristics, skip_perplexity=skip_perplexity)

    return {
        "filename":                  safe_name,
        "is_ai_generated":           result["is_ai_generated"],
        "label":                     result.get("label", "human_written"),
        "confidence":                result["confidence"],
        "explanation":               result["reasons"],
        "matched_ai_phrases":        heuristics.get("matched_ai_phrases", []),
        "matched_template_patterns": heuristics.get("matched_template_patterns", []),
        "raw_heuristics":            heuristics,
        "features":                  result.get("features", {}),
        "feature_importances":       result.get("feature_importances", {}),
        "model_results":             result.get("model_results", {}),
        "debug_info": {
            "extracted_text_preview":    (text[:500] + "...") if len(text) > 500 else text,
            "preprocessed_text_preview": (clean_text[:500] + "...") if len(clean_text) > 500 else clean_text,
        },
    }
