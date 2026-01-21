import os
from pypdf import PdfReader
from docx import Document

class TextExtractor:
    @staticmethod
    def extract_from_file(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return TextExtractor.extract_from_pdf(file_path)
        elif ext == ".docx":
            return TextExtractor.extract_from_docx(file_path)
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    @staticmethod
    def extract_from_pdf(file_path: str) -> str:
        reader = PdfReader(file_path)
        return "\n".join([page.extract_text() for page in reader.pages])

    @staticmethod
    def extract_from_docx(file_path: str) -> str:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
