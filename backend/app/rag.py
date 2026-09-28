from __future__ import annotations

from typing import Protocol

from backend.app import config
from backend.app.retrieval import DocumentIndex, SearchResult
from backend.app.schemas.chat import ChatHistoryMessage


NO_ANSWER = "The processed documents do not contain enough information to answer that question."
HISTORY_MESSAGES = 6


class LanguageModel(Protocol):
    def generate(self, prompt: str) -> str: ...


def answer_question(
    index: DocumentIndex,
    llm: LanguageModel,
    session_id: str,
    question: str,
    history: list[ChatHistoryMessage],
) -> tuple[str, list[SearchResult]]:
    results = index.search(session_id, build_search_query(question, history), config.TOP_K)
    if not results:
        return NO_ANSWER, []

    answer = llm.generate(build_prompt(question, results, history)).strip()
    if not answer or answer.startswith(NO_ANSWER):
        return NO_ANSWER, []
    return answer, results


def build_search_query(question: str, history: list[ChatHistoryMessage]) -> str:
    previous_questions = [message.content for message in history if message.role == "user"]
    return " ".join(previous_questions[-1:] + [question])


def build_prompt(
    question: str,
    results: list[SearchResult],
    history: list[ChatHistoryMessage],
) -> str:
    context = "\n\n".join(
        f"[{result.chunk.filename}, page {result.chunk.page or 'n/a'}]\n{result.chunk.text}"
        for result in results
    )
    conversation = "\n".join(
        f"{message.role}: {message.content}" for message in history[-HISTORY_MESSAGES:]
    )
    return (
        "You answer questions about the user's documents.\n"
        "Answer using only the context below. Be direct and concise.\n"
        "Use the conversation only to understand follow-up questions.\n"
        "Do not mention sources, chunks or page numbers; the UI shows them separately.\n"
        f"If the context does not contain the answer, reply exactly: {NO_ANSWER}\n\n"
        f"Conversation:\n{conversation or 'None'}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
