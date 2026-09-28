from typing import Literal, Optional

from pydantic import BaseModel


class ProcessedDocument(BaseModel):
    filename: str
    status: Literal["ready", "failed"]
    chunk_count: int
    error: Optional[str] = None


class ProcessDocumentsResponse(BaseModel):
    documents: list[ProcessedDocument]
