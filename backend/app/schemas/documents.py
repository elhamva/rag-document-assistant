from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ProcessedDocument(BaseModel):
    id: str
    filename: str
    status: str
    chunk_count: int
    error: Optional[str] = None


class ProcessDocumentsResponse(BaseModel):
    documents: list[ProcessedDocument]
