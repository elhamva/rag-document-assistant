from __future__ import annotations

import logging
from typing import Optional, Protocol

from backend.app.schemas.chat import ChatHistoryMessage


logger = logging.getLogger(__name__)

REWRITE_TIMEOUT_SECONDS = 20
MAX_QUERY_LENGTH = 300


class RewriteModel(Protocol):
    def generate(self, prompt: str, timeout: float) -> str: ...


def rewrite_query(llm: RewriteModel, question: str, history: list[ChatHistoryMessage]) -> str:
    """Turn a follow-up question into a standalone search query.

    Only used for retrieval; the answer prompt still gets the original question and history.
    Falls back to the original question whenever the rewrite cannot be trusted.
    """
    previous_question = last_message(history, "user")
    if previous_question is None:
        return question

    prompt = build_rewrite_prompt(previous_question, last_message(history, "assistant") or "", question)
    try:
        output = llm.generate(prompt, timeout=REWRITE_TIMEOUT_SECONDS)
    except Exception:  # the rewrite is optional, so it must never break retrieval
        logger.warning("Query rewrite failed; using the original question.", exc_info=True)
        return question

    rewritten = clean_output(output)
    if not is_valid_rewrite(question, rewritten):
        logger.warning("Query rewrite output rejected: %r", output)
        return question

    logger.info("Search query: %r -> %r", question, rewritten)
    return rewritten


def last_message(history: list[ChatHistoryMessage], role: str) -> Optional[str]:
    for message in reversed(history):
        if message.role == role and message.content.strip():
            return message.content.strip()
    return None


def build_rewrite_prompt(previous_question: str, previous_answer: str, question: str) -> str:
    return (
        "Rewrite the current question as a standalone search query.\n"
        "Rules:\n"
        "- Resolve pronouns and implicit references using the previous exchange when necessary.\n"
        "- Preserve exact names, identifiers, numbers and model names.\n"
        "- If the current question is already standalone or changes topic, return it unchanged.\n"
        "- Output only the rewritten query, with no explanation.\n\n"
        f"Previous question: {previous_question}\n"
        f"Previous answer: {previous_answer}\n"
        f"Current question: {question}\n"
        "Standalone query:"
    )


def clean_output(output: object) -> str:
    if not isinstance(output, str):
        return ""
    return output.strip().strip("\"'`").strip()


def is_valid_rewrite(question: str, rewritten: str) -> bool:
    if not rewritten or "\n" in rewritten or len(rewritten) > MAX_QUERY_LENGTH:
        return False
    # Exact identifiers such as "llama3.2:3b" or "HB-150" must survive the rewrite untouched.
    return all(identifier in rewritten for identifier in identifiers(question))


def identifiers(text: str) -> list[str]:
    tokens = (token.strip("?!.,;:()\"'") for token in text.split())
    return [token for token in tokens if any(character.isdigit() for character in token)]
