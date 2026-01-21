from services.features import FeatureExtractor

class ResumeModel:
    def __init__(self):
        # Load model here (e.g., pickle or from_pretrained)
        pass

    def predict(self, text: str, heuristics: dict) -> dict:
        # 1. Extract Advanced Features
        features = FeatureExtractor.extract_features(text)
        
        # 2. Combine with Heuristics
        score = 0.0
        details = []

        if heuristics.get("uses_ai_phrases"):
            score += 0.4
            details.append("Contains common AI-generated phrases.")

        if heuristics.get("has_template_structure"):
            score += 0.2
            details.append("Follows a generic AI/Template structure.")

        # Random/Features logic would go here
        
        # Feature-based scoring (Simple Logic for now)
        if features.get("lexical_diversity", 0.5) < 0.4:
            score += 0.2
            details.append("Low lexical diversity (repetitive vocabulary).")
            
        if features.get("sent_len_std", 10) < 5:
             score += 0.1
             details.append("Very uniform sentence lengths (robotic flow).")

        # Perplexity Check (AI texts often have low perplexity < 30-50)
        ppl = features.get("perplexity", 100)
        if ppl > 0 and ppl < 40:
            score += 0.3
            details.append(f"Low Perplexity ({int(ppl)}): Indicative of AI text.")
        elif ppl > 80:
            score -= 0.1
            details.append(f"High Perplexity ({int(ppl)}): Indicative of Human text.")

        # Normalize score
        confidence = min(max(score, 0.0), 1.0)
        
        return {
            "is_ai_generated": confidence > 0.5,
            "confidence": confidence,
            "reasons": details,
            "features": features
        }
