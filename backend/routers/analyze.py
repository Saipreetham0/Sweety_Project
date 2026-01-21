from services.extraction import TextExtractor
from services.preprocessing import TextPreprocessor
from services.weak_supervision import WeakSupervision
from services.model import ResumeModel
import os

router = APIRouter(prefix="/analyze", tags=["analyze"])
UPLOAD_DIR = "uploads"
model = ResumeModel()

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
    
    # 3. Weak Supervision (Heuristics)
    heuristics = WeakSupervision.apply_heuristics(clean_text)

    # 4. Hybrid ML Prediction
    result = model.predict(clean_text, heuristics)

    return {
        "filename": filename,
        "is_ai_generated": result["is_ai_generated"],
        "confidence": result["confidence"],
        "explanation": result["reasons"],
        "raw_heuristics": heuristics
    }
