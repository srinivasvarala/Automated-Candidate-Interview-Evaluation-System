"""Extract text from uploaded resume files (PDF or plain text)."""

import io
import logging

from pypdf import PdfReader

logger = logging.getLogger(__name__)

MAX_RESUME_CHARS = 8000  # truncate to keep prompts manageable


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file's bytes."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    full_text = "\n".join(pages).strip()
    if not full_text:
        raise ValueError("Could not extract any text from the PDF. The file may be image-based.")
    return full_text[:MAX_RESUME_CHARS]


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract text from a resume file. Supports PDF and plain text."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    if lower.endswith((".txt", ".md", ".text")):
        return file_bytes.decode("utf-8", errors="replace").strip()[:MAX_RESUME_CHARS]
    raise ValueError(f"Unsupported file type: {filename}. Please upload a PDF or text file.")
