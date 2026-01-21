class ResumeModel:
    def __init__(self):
        # Load model here (e.g., pickle or from_pretrained)
        pass

    def predict(self, text: str, heuristics: dict) -> dict:
        # Combine heuristics and model prediction
        # For prototype: strict heuristic weight or mock score
        
        score = 0.0
        details = []

        if heuristics.get("uses_ai_phrases"):
            score += 0.4
            details.append("Contains common AI-generated phrases.")

        # Random/Features logic would go here
        
        # Normalize score
        confidence = min(max(score, 0.0), 1.0)
        
        return {
            "is_ai_generated": confidence > 0.5,
            "confidence": confidence,
            "reasons": details
        }
