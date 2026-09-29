import json
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from backend.app.dependencies import get_index, get_llm
from backend.app.main import app
from backend.app.ollama import OllamaError
from backend.app.reranker import NoReranker
from backend.app.retrieval import DocumentIndex


TOPICS = {
    "vehicle": ("car", "automobile", "vehicle"),
    "fruit": ("banana", "fruit"),
    "warranty": ("warranty", "guarantee"),
}


class FakeEmbedder:
    def __init__(self) -> None:
        self.embedded_texts: list[str] = []
        self.available = True

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self.available:
            raise OllamaError("Ollama returned HTTP 404: model not found")
        self.embedded_texts.extend(texts)
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        text = text.lower()
        return [float(any(word in text for word in words)) for words in TOPICS.values()]


class FakeLLM:
    def __init__(self) -> None:
        self.reply = json.dumps({"answerable": True, "answer": "A grounded answer."})
        self.rewritten_query = ""  # empty output makes the rewriter fall back to the question
        self.prompts: list[str] = []
        self.rewrite_prompts: list[str] = []
        self.model = "default-test-model"
        self.model_overrides: list[str] = []
        self.available = True

    def answers(self, answer: str, answerable: bool = True) -> None:
        self.reply = json.dumps({"answerable": answerable, "answer": answer})

    def generate(self, prompt: str, timeout: float = 120, json_schema: Optional[dict] = None) -> str:
        if not self.available:
            raise OllamaError("Cannot reach Ollama at http://localhost:11434.")
        if prompt.startswith("Rewrite the current question"):
            self.rewrite_prompts.append(prompt)
            return self.rewritten_query
        self.prompts.append(prompt)
        return self.reply

    def with_model(self, model: str) -> "FakeLLM":
        self.model = model
        self.model_overrides.append(model)
        return self


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def index(embedder: FakeEmbedder) -> DocumentIndex:
    return DocumentIndex(embedder=embedder, reranker=NoReranker())


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(index: DocumentIndex, llm: FakeLLM):
    app.dependency_overrides[get_index] = lambda: index
    app.dependency_overrides[get_llm] = lambda: llm
    yield TestClient(app)
    app.dependency_overrides.clear()
