import sys
import os

_CURR_DIR = os.path.abspath(os.path.dirname(__file__))
_BACKEND_DIR = os.path.join(_CURR_DIR, "backend")

for p in [_CURR_DIR, _BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    print("Verifying backend services...")
    app = None
    try:
        from backend.main import app
    except ImportError:
        try:
            from main import app
        except Exception as e:
            print(f"Note: FastAPI app not importable directly in system python: {e}")

    from services.extraction import TextExtractor
    from services.preprocessing import TextPreprocessor
    from services.weak_supervision import WeakSupervision
    from services.model import ResumeModel

    print("Core service imports successful.")

    # Test heuristics
    text = "This text is meticulously crafted to delve into the complex landscape of technology."
    ws = WeakSupervision()
    heuristics = ws.apply_heuristics(text)
    print(f"Heuristics test: matched {len(heuristics['matched_ai_phrases'])} phrases: {heuristics['matched_ai_phrases']}")

    if heuristics["uses_ai_phrases"]:
        print("SUCCESS: AI phrase detected.")
    else:
        print("FAILURE: AI phrase NOT detected.")
        sys.exit(1)

    # Test model instantiation
    model = ResumeModel()
    print("Model instantiated.")

    # Test prediction
    pred = model.predict(text, heuristics, skip_perplexity=True)
    print(f"Prediction successful: label={pred.get('label')}, confidence={pred.get('confidence')}")

    # Path traversal must not escape uploads/ (regression check)
    from fastapi import HTTPException
    from routers.analyze import analyze_resume
    try:
        analyze_resume("../../verify_setup.py", True)
        print("FAILURE: path traversal not blocked.")
        sys.exit(1)
    except HTTPException as e:
        assert e.status_code == 404, e.status_code
    print("Path traversal blocked.")

    # Bundled sample used by the frontend Quick Test button resolves
    assert analyze_resume("AI_Candidate_1.txt", True)["label"]
    print("Sample corpus lookup OK.")

    print("All verifications passed!")

except Exception as e:
    print(f"Verification failed: {e}")
    sys.exit(1)
