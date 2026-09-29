from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

from pypdf import PdfReader


CHUNK_WORDS = 180
CHUNK_OVERLAP_WORDS = 40
TEXT_EXTENSIONS = {".txt", ".md"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {".pdf"}


class DocumentError(ValueError):
    pass


@dataclass
class Chunk:
    id: str
    filename: str
    page: Optional[int]
    text: str
    embedding: Optional[list[float]] = None


def content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def parse_document(
    filename: str,
    content: bytes,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[Chunk]:
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        pages = read_pdf(content)
    elif extension in TEXT_EXTENSIONS:
        pages = [(None, decode_text(content))]
    else:
        raise DocumentError("Unsupported file type. Upload PDF, TXT or MD files.")

    document_id = content_hash(content)
    chunks = []
    for page, text in pages:
        for piece in split_text(text, chunk_words, overlap_words):
            chunks.append(
                Chunk(
                    id=f"{document_id}-{len(chunks)}",
                    filename=filename,
                    page=page,
                    text=piece,
                )
            )

    if not chunks:
        raise DocumentError("No readable text found. Scanned PDFs are not supported.")
    return chunks


def read_pdf(content: bytes) -> list[tuple[int, str]]:
    try:
        reader = PdfReader(BytesIO(content))
        return [
            (number, page.extract_text() or "")
            for number, page in enumerate(reader.pages, start=1)
        ]
    except Exception as exc:
        raise DocumentError("Could not read the PDF.") from exc


def decode_text(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def split_text(
    text: str, chunk_words: int = CHUNK_WORDS, overlap_words: int = CHUNK_OVERLAP_WORDS
) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + chunk_words]))
        if start + chunk_words >= len(words):
            break
        start += chunk_words - overlap_words
    return chunks
