import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

try:
    print("Verifying backend imports...")
    from backend.main import app
    from backend.services.extraction import TextExtractor
    from backend.services.preprocessing import TextPreprocessor
    from backend.services.weak_supervision import WeakSupervision
    from backend.services.model import ResumeModel
    
    print("Imports successful.")
    
    # Test model instantiation
    model = ResumeModel()
    print("Model instantiated.")
    
    # Test heuristics
    text = "This text is meticulously crafted to delve into the landscape of AI."
    heuristics = WeakSupervision.apply_heuristics(text)
    print(f"Heuristics test: {heuristics}")
    
    if heuristics['uses_ai_phrases']:
        print("SUCCESS: AI phrase detected.")
    else:
        print("FAILURE: AI phrase NOT detected.")
        sys.exit(1)

    print("Verification passed!")

except Exception as e:
    print(f"Verification failed: {e}")
    sys.exit(1)
