class WeakSupervision:
    @staticmethod
    def apply_heuristics(text: str) -> dict:
        heuristics = {
            "uses_ai_phrases": WeakSupervision._check_ai_phrases(text),
            "has_template_structure": WeakSupervision._check_structure(text),
            "perfect_grammar": False, # Placeholder
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

    @staticmethod
    def _check_structure(text: str) -> bool:
        # Check for standard template headers in a typical order
        # Many AI models output: Summary -> Skills -> Experience -> Education
        text_lower = text.lower()
        required_sections = [
            ["professional summary", "summary", "objective"],
            ["skills", "core competencies", "technical skills"],
            ["experience", "work history", "employment"],
            ["education", "academic background"]
        ]
        
        found_count = 0
        for section_group in required_sections:
            if any(header in text_lower for header in section_group):
                found_count += 1
        
        # If all 4 key sections are present, it might be a standard template (weak signal)
        return found_count == 4
