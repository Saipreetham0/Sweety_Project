# MODIFIED
"""
POST /analyze/{filename}

Runs the full 4-step pipeline:
  1. Text extraction
  2. Preprocessing
  3. Weak supervision (heuristics from Dataset B)
  4. Hybrid RF prediction (model trained on Dataset A)

Returns an extended response including label (3-class), matched phrases,
and feature importances. Duplicate "explanation" key bug from original is fixed.
"""

from fastapi import APIRouter, HTTPException
import os

from services.extraction import TextExtractor
from services.preprocessing import TextPreprocessor
from services.weak_supervision import WeakSupervision
from services.model import ResumeModel

router     = APIRouter(prefix="/analyze", tags=["analyze"])
UPLOAD_DIR = "uploads"

# Singletons — load once at startup so model weights are not reloaded per request
_model = ResumeModel()
_ws    = WeakSupervision()


@router.post("/{filename}")
async def analyze_resume(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # 1. Extraction
    try:
        text = TextExtractor.extract_from_file(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Extraction failed: {str(e)}")

    # 2. Preprocessing
    clean_text = TextPreprocessor.clean_text(text)

    # 3. Weak Supervision
    heuristics = _ws.apply_heuristics(clean_text)

    # 4. Hybrid ML Prediction
    result = _model.predict(clean_text, heuristics)

    return {
        "filename":                  filename,
        "is_ai_generated":           result["is_ai_generated"],
        "label":                     result.get("label", "human_written"),
        "confidence":                result["confidence"],
        "explanation":               result["reasons"],   # single key — duplicate bug fixed
        "matched_ai_phrases":        heuristics.get("matched_ai_phrases", []),
        "matched_template_patterns": heuristics.get("matched_template_patterns", []),
        "raw_heuristics":            heuristics,
        "features":                  result.get("features", {}),
        "feature_importances":       result.get("feature_importances", {}),
        "debug_info": {
            "extracted_text_preview":    (text[:500] + "...") if len(text) > 500 else text,
            "preprocessed_text_preview": (clean_text[:500] + "...") if len(clean_text) > 500 else clean_text,
        },
    }
