from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, StringConstraints


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    session_id: str = "default"
    history: list[ChatHistoryMessage] = Field(default_factory=list)
    model: Optional[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]] = None


class ChatSource(BaseModel):
    filename: str
    page: Optional[int] = None
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
