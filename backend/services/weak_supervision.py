class WeakSupervision:
    @staticmethod
    def apply_heuristics(text: str) -> dict:
        heuristics = {
            "uses_ai_phrases": WeakSupervision._check_ai_phrases(text),
            "perfect_grammar": False, # Placeholder
            "uniform_structure": False # Placeholder
        }
        return heuristics

    @staticmethod
    def _check_ai_phrases(text: str) -> bool:
        ai_phrases = [
            "delve into",
            "testament to",
            "landscape of",
            "meticulously crafted",
            "tapestry of",
            "underscores the importance",
            "poised to",
            "leveraging the power of",
            "realm of",
            "navigate the complexities",
            "foster a culture of"
            # Add more known AI-isms
        ]
        text_lower = text.lower()
        for phrase in ai_phrases:
            if phrase in text_lower:
                return True
        return False
