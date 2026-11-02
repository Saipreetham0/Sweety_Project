import os

class TextExtractor:
    @staticmethod
    def extract_from_file(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return TextExtractor.extract_from_pdf(file_path)
        elif ext == ".docx":
            return TextExtractor.extract_from_docx(file_path)
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    @staticmethod
    def extract_from_pdf(file_path: str) -> str:
        try:
            from pypdf import PdfReader  # type: ignore[import]
        except ImportError:
            raise RuntimeError("pypdf is required for PDF extraction. Please install pypdf.")
        reader = PdfReader(file_path)
        pages_text = []
        for page in reader.pages:
            # Use layout extraction mode to preserve horizontal word grouping and layout
            try:
                t = page.extract_text(extraction_mode="layout")
            except Exception:
                t = page.extract_text()
            if t:
                pages_text.append(t)
        return "\n".join(pages_text)

    @staticmethod
    def extract_from_docx(file_path: str) -> str:
        try:
            from docx import Document  # type: ignore[import]
        except ImportError:
            raise RuntimeError("python-docx is required for DOCX extraction. Please install python-docx.")
        doc = Document(file_path)
        parts = [para.text for para in doc.paragraphs if para.text]
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                deduped = []
                for c in cells:
                    if not deduped or c != deduped[-1]:
                        deduped.append(c)
                if deduped:
                    parts.append(" | ".join(deduped))
        return "\n".join(parts)
