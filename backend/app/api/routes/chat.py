from fastapi import APIRouter, Depends, HTTPException

from backend.app.dependencies import get_index, get_llm
from backend.app.ollama import OllamaClient, OllamaError
from backend.app.rag import answer_question
from backend.app.retrieval import DocumentIndex
from backend.app.schemas.chat import ChatRequest, ChatResponse, ChatSource


router = APIRouter(prefix="/chat")


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    index: DocumentIndex = Depends(get_index),
    llm: OllamaClient = Depends(get_llm),
) -> ChatResponse:
    answer_llm = llm.with_model(request.model) if request.model else llm
    try:
        answer, results = answer_question(
            index, answer_llm, request.session_id, request.question, request.history
        )
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    sources = [
        ChatSource(filename=result.chunk.filename, page=result.chunk.page, text=result.chunk.text)
        for result in results
    ]
    return ChatResponse(answer=answer, sources=sources)
