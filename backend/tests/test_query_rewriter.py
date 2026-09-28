from typing import Optional

from backend.app.ollama import OllamaError
from backend.app.query_rewriter import REWRITE_TIMEOUT_SECONDS, rewrite_query
from backend.app.schemas.chat import ChatHistoryMessage


class StubModel:
    def __init__(self, output: str = "", error: Optional[Exception] = None) -> None:
        self.output = output
        self.error = error
        self.calls: list[tuple[str, float]] = []

    def generate(self, prompt: str, timeout: float) -> str:
        self.calls.append((prompt, timeout))
        if self.error:
            raise self.error
        return self.output


def exchange(question: str, answer: str) -> list[ChatHistoryMessage]:
    return [
        ChatHistoryMessage(role="user", content=question),
        ChatHistoryMessage(role="assistant", content=answer),
    ]


NORTHSTAR = exchange(
    "Who is responsible for the technical side of Project Northstar?",
    "Marco Stein is the technical lead.",
)


def test_no_history_returns_question_without_calling_the_model() -> None:
    model = StubModel("something else")

    assert rewrite_query(model, "What is Project Northstar?", []) == "What is Project Northstar?"
    assert model.calls == []


def test_pronoun_follow_up_becomes_standalone_query() -> None:
    model = StubModel("Where did Marco Stein complete his master's degree?")

    query = rewrite_query(model, "Where did he complete his master's degree?", NORTHSTAR)

    assert query == "Where did Marco Stein complete his master's degree?"
    prompt, timeout = model.calls[0]
    assert "Previous question: Who is responsible for the technical side of Project Northstar?" in prompt
    assert "Previous answer: Marco Stein is the technical lead." in prompt
    assert "Current question: Where did he complete his master's degree?" in prompt
    assert timeout == REWRITE_TIMEOUT_SECONDS


def test_only_the_latest_exchange_is_sent_to_the_model() -> None:
    model = StubModel("Where did Marco Stein complete his master's degree?")
    history = exchange("What does Aurora Logistics do?", "It is a logistics company.") + NORTHSTAR

    rewrite_query(model, "Where did he complete his master's degree?", history)

    assert "Aurora Logistics" not in model.calls[0][0]


def test_topic_change_stays_unchanged() -> None:
    model = StubModel("What is llama3.2:3b used for?")
    history = exchange("Where did he complete his master's degree?", "The Technical University of Munich.")

    assert rewrite_query(model, "What is llama3.2:3b used for?", history) == "What is llama3.2:3b used for?"


def test_model_failure_falls_back_to_question() -> None:
    model = StubModel(error=OllamaError("Cannot reach Ollama at http://localhost:11434."))

    assert rewrite_query(model, "Where did he study?", NORTHSTAR) == "Where did he study?"


def test_empty_or_invalid_output_falls_back_to_question() -> None:
    for output in ["", "   ", "Marco Stein.\nHe studied in Munich.", "x" * 500]:
        assert rewrite_query(StubModel(output), "Where did he study?", NORTHSTAR) == "Where did he study?"


def test_output_quotes_are_stripped() -> None:
    model = StubModel('"Where did Marco Stein study?"')

    assert rewrite_query(model, "Where did he study?", NORTHSTAR) == "Where did Marco Stein study?"


def test_rewrite_that_changes_an_identifier_is_rejected() -> None:
    model = StubModel("What is llama 3.2 used for in Project Northstar?")
    history = exchange("Who leads Project Northstar?", "Dr. Lena Hoffmann.")

    assert rewrite_query(model, "What is llama3.2:3b used for?", history) == "What is llama3.2:3b used for?"
