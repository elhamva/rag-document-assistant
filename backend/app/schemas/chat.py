from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChatHistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"
    history: list[ChatHistoryMessage] = Field(default_factory=list)


class ChatSource(BaseModel):
    chunk_id: str
    filename: str
    page: Optional[int] = None
    snippet: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
