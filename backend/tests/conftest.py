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
    def __init__(self, answer: str = "A grounded answer.") -> None:
        self.answer = answer
        self.prompts: list[str] = []
        self.available = True

    def generate(self, prompt: str) -> str:
        if not self.available:
            raise OllamaError("Cannot reach Ollama at http://localhost:11434.")
        self.prompts.append(prompt)
        return self.answer


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
