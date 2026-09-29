from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional, Protocol

from backend.app import config
from backend.app.query_rewriter import rewrite_query
from backend.app.retrieval import DocumentIndex, SearchResult
from backend.app.schemas.chat import ChatHistoryMessage


logger = logging.getLogger(__name__)

NO_ANSWER = "The processed documents do not contain enough information to answer that question."
HISTORY_MESSAGES = 6

# The model answers in this shape, so a refusal is a flag rather than a sentence we have to recognise.
# "answerable" comes first: deciding before writing refused 15 of 15 unanswerable questions in
# backend/scripts/evaluate_answers.py, against 12 of 15 with "answer" first (once written, it gets trusted).
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
    },
    "required": ["answerable", "answer"],
}


class LanguageModel(Protocol):
    def generate(self, prompt: str, timeout: float = ..., json_schema: Optional[dict] = ...) -> str: ...


@dataclass
class ModelReply:
    answer: str
    answerable: bool


def answer_question(
    index: DocumentIndex,
    llm: LanguageModel,
    session_id: str,
    question: str,
    history: list[ChatHistoryMessage],
) -> tuple[str, list[SearchResult]]:
    results = index.search(session_id, rewrite_query(llm, question, history), config.TOP_K)
    if not results:
        return NO_ANSWER, []

    reply = parse_reply(llm.generate(build_prompt(question, results, history), json_schema=ANSWER_SCHEMA))
    if reply is None or not reply.answerable or not reply.answer:
        return NO_ANSWER, []
    return reply.answer, results


def parse_reply(output: str) -> Optional[ModelReply]:
    try:
        data = json.loads(output)
        return ModelReply(
            answer=str(data["answer"]).strip(),
            answerable=data["answerable"] is True,
        )
    except (ValueError, KeyError, TypeError):
        logger.warning("Model reply is not valid JSON: %r", output)
        return None


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
        "Use only the context below. Use the conversation only to understand follow-up questions.\n"
        "Reply with a JSON object:\n"
        '- "answerable": true only if the context contains the answer; false if it does not, '
        "or only partly.\n"
        '- "answer": a direct, concise answer. Do not mention sources, chunks or page numbers; '
        "the UI shows them separately.\n\n"
        f"Conversation:\n{conversation or 'None'}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
