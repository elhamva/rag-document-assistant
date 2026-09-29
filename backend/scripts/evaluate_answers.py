"""End-to-end refusal check: does the app answer what the handbook covers and refuse what it does not?

Runs the real pipeline (retrieval, reranker, chat model) against the handbook. Needs Ollama.
"""

from __future__ import annotations

from eval_common import HANDBOOK, handbook_questions, unanswerable_questions

from backend.app.ingestion import parse_document
from backend.app.ollama import OllamaClient
from backend.app.rag import NO_ANSWER, answer_question
from backend.app.reranker import build_reranker
from backend.app.retrieval import DocumentIndex

SESSION = "evaluation"


def main() -> None:
    ollama = OllamaClient()
    index = DocumentIndex(embedder=ollama, reranker=build_reranker())
    index.replace(SESSION, {"handbook": parse_document(HANDBOOK.name, HANDBOOK.read_bytes())})

    answerable = handbook_questions()
    false_refusals, sources_shown, gold_shown = [], 0, 0
    for question in answerable:
        answer, results = answer_question(index, ollama, SESSION, question.text, [])
        if answer == NO_ANSWER:
            false_refusals.append(question.text)
            continue
        sources_shown += len(results)
        gold_shown += any(question.expected.lower() in result.chunk.text.lower() for result in results)

    unanswerable = unanswerable_questions()
    not_refused = []
    for question in unanswerable:
        answer, results = answer_question(index, ollama, SESSION, question, [])
        if answer != NO_ANSWER:
            not_refused.append((question, answer, len(results)))

    answered = len(answerable) - len(false_refusals)
    print(f"Model: {ollama.model}\n")
    print(f"Answerable questions:   {len(answerable)}")
    print(f"  answered              {answered}")
    print(f"  wrongly refused       {len(false_refusals)}")
    print(f"  right passage shown   {gold_shown} of {answered}")
    print(f"  sources per answer    {sources_shown / max(answered, 1):.1f}")
    print(f"Unanswerable questions: {len(unanswerable)}")
    print(f"  refused               {len(unanswerable) - len(not_refused)}")
    print(f"  answered anyway       {len(not_refused)}")

    for question in false_refusals:
        print(f"\nWrongly refused: {question}")
    for question, answer, source_count in not_refused:
        print(f"\nNot refused ({source_count} sources shown): {question}\n  -> {answer}")


if __name__ == "__main__":
    main()
