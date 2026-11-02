import re

class TextPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""

        # Remove non-printable characters while preserving newlines and tabs
        text = "".join(c for c in text if c.isprintable() or c in "\r\n\t")

        # Standardize carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse horizontal whitespace per line (preserving structural newlines)
        lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Collapse 3 or more consecutive blank lines down to 2
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        return text
