from __future__ import annotations

import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException

from backend.app.document_index import SearchResult, document_index
from backend.app.schemas.chat import ChatHistoryMessage, ChatRequest, ChatResponse, ChatSource


NO_CONTEXT_ANSWER = (
    "The processed documents do not contain enough information to answer that question."
)
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_RETRIEVAL_LIMIT = 4
OLLAMA_BASE_URL_ENV_VAR = "OLLAMA_BASE_URL"
OLLAMA_MODEL_ENV_VAR = "OLLAMA_MODEL"
RETRIEVAL_LIMIT_ENV_VAR = "RAG_TOP_K"

router = APIRouter(prefix="/chat")


@router.post("", response_model=ChatResponse)
def answer_question(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    retrieval_query = build_retrieval_query(question, request.history)
    results = document_index.search_with_scores(
        retrieval_query,
        limit=get_retrieval_limit(),
        session_id=request.session_id,
    )

    if not question or not results:
        return ChatResponse(answer=NO_CONTEXT_ANSWER, sources=[])

    try:
        answer = generate_grounded_answer(question, results, request.history)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ChatResponse(
        answer=answer,
        sources=[build_source(result) for result in results],
    )


class OllamaError(RuntimeError):
    pass


def get_retrieval_limit() -> int:
    value = os.getenv(RETRIEVAL_LIMIT_ENV_VAR)
    if value is None:
        return DEFAULT_RETRIEVAL_LIMIT

    try:
        return max(1, int(value))
    except ValueError:
        return DEFAULT_RETRIEVAL_LIMIT


def build_retrieval_query(question: str, history: list[ChatHistoryMessage]) -> str:
    recent_user_messages = [
        message.content.strip()
        for message in history[-6:]
        if message.role == "user" and message.content.strip()
    ]
    return " ".join([*recent_user_messages, question])


def generate_grounded_answer(
    question: str,
    results: list[SearchResult],
    history: list[ChatHistoryMessage] | None = None,
) -> str:
    context = build_context(results)
    if not context:
        return NO_CONTEXT_ANSWER

    prompt = build_prompt(question, context, history or [])
    return call_ollama(prompt).strip() or NO_CONTEXT_ANSWER


def build_context(results: list[SearchResult]) -> str:
    context_parts = []

    for index, result in enumerate(results, start=1):
        page = (
            result.chunk.page_number
            if result.chunk.page_number is not None
            else "unknown"
        )
        context_parts.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"Filename: {result.chunk.filename}",
                    f"Page: {page}",
                    f"Chunk ID: {result.chunk.id}",
                    f"Text: {result.chunk.text}",
                ]
            )
        )

    return "\n\n".join(context_parts)


def build_prompt(
    question: str,
    context: str,
    history: list[ChatHistoryMessage],
) -> str:
    history_text = build_history_text(history)
    return (
        "You are a document question-answering assistant.\n"
        "Answer using only the supplied context.\n"
        "Answer directly and concisely.\n"
        "Use the recent conversation only to understand follow-up questions.\n"
        "Do not say \"According to Source X\" or mention numbered sources.\n"
        "Do not invent citation numbers or reference source indices in the answer.\n"
        "The UI displays sources separately.\n"
        "If the context does not contain enough information, say exactly: "
        f"{NO_CONTEXT_ANSWER}\n"
        "Do not use outside knowledge.\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def build_history_text(history: list[ChatHistoryMessage]) -> str:
    lines = [
        f"{message.role}: {message.content.strip()}"
        for message in history[-6:]
        if message.role in {"user", "assistant"} and message.content.strip()
    ]
    return "\n".join(lines) if lines else "None"


def call_ollama(prompt: str) -> str:
    base_url = os.getenv(OLLAMA_BASE_URL_ENV_VAR, DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    model = os.getenv(OLLAMA_MODEL_ENV_VAR, DEFAULT_OLLAMA_MODEL)
    request = Request(
        f"{base_url}/api/generate",
        data=json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise OllamaError(format_ollama_http_error(exc)) from exc
    except (OSError, URLError, json.JSONDecodeError) as exc:
        raise OllamaError(
            "Ollama is not available. Start Ollama and pull the configured model."
        ) from exc

    answer = payload.get("response")
    if not isinstance(answer, str):
        raise OllamaError("Ollama returned an invalid response.")

    return answer


def format_ollama_http_error(exc: HTTPError) -> str:
    try:
        payload = json.loads(exc.read().decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        payload = None

    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return f"Ollama returned HTTP {exc.code}: {payload['error']}"

    return f"Ollama returned HTTP {exc.code}."


def build_source(result: SearchResult) -> ChatSource:
    return ChatSource(
        chunk_id=result.chunk.id,
        filename=result.chunk.filename,
        page=result.chunk.page_number,
        snippet=make_snippet(result.chunk.text),
        score=result.score,
    )


def make_snippet(text: str, max_length: int = 280) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_length:
        return text

    return f"{text[: max_length - 1].rstrip()}..."
