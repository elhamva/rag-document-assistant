import json

from io import BytesIO
from urllib.error import HTTPError, URLError

from fastapi.testclient import TestClient

from backend.app.api.routes import chat
from backend.app.document_index import (
    EmbeddingError,
    EmbeddingProvider,
    Reranker,
    SearchResult,
    document_index,
    process_document,
)
from backend.app.main import app


client = TestClient(app)


class FailingEmbeddingProvider(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("Embeddings unavailable.")


class PassthroughReranker(Reranker):
    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        return candidates


def setup_function() -> None:
    document_index._embedding_provider = FailingEmbeddingProvider()
    document_index._reranker = PassthroughReranker()
    document_index.clear()


class FakeOllamaResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "FakeOllamaResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_chat_returns_llm_answer_and_real_sources(monkeypatch) -> None:
    chunks = process_document(
        "guide.txt",
        (
            b"Retrieval augmented generation uses indexed document chunks. "
            b"Those chunks keep citation metadata for later answers."
        ),
    )
    document_index.replace_chunks(chunks)
    prompts = []

    def fake_call_ollama(prompt: str) -> str:
        prompts.append(prompt)
        return "Citation metadata is kept for later answers."

    monkeypatch.setattr(chat, "call_ollama", fake_call_ollama)

    response = client.post("/chat", json={"question": "citation metadata"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Citation metadata is kept for later answers."
    assert payload["sources"][0]["chunk_id"] == chunks[0].id
    assert payload["sources"][0]["filename"] == "guide.txt"
    assert payload["sources"][0]["page"] is None
    assert payload["sources"][0]["snippet"] == chunks[0].text
    assert payload["sources"][0]["score"] > 0
    assert "Question: citation metadata" in prompts[0]
    assert chunks[0].text in prompts[0]
    assert "Answer using only the supplied context." in prompts[0]


def test_chat_uses_history_for_follow_up_retrieval(monkeypatch) -> None:
    chunks = process_document(
        "profile.txt",
        b"Hoa earned a Master of Science degree in computer science.",
    )
    document_index.replace_chunks(chunks)
    prompts = []

    def fake_call_ollama(prompt: str) -> str:
        prompts.append(prompt)
        return "Hoa earned a Master of Science degree."

    monkeypatch.setattr(chat, "call_ollama", fake_call_ollama)

    response = client.post(
        "/chat",
        json={
            "question": "what degree?",
            "history": [{"role": "user", "content": "Tell me about Hoa"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Hoa earned a Master of Science degree."
    assert payload["sources"][0]["chunk_id"] == chunks[0].id
    assert "user: Tell me about Hoa" in prompts[0]


def test_chat_returns_clear_answer_when_no_context_matches() -> None:
    chunks = process_document(
        "guide.txt",
        b"This document only describes citation metadata.",
    )
    document_index.replace_chunks(chunks)

    response = client.post("/chat", json={"question": "deployment queue"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": (
            "The processed documents do not contain enough information to answer "
            "that question."
        ),
        "sources": [],
    }


def test_process_then_chat_returns_answer_and_sources(monkeypatch) -> None:
    content = b"Hoa earned a Master of Science degree in computer science."
    process_response = client.post(
        "/documents/process",
        files=[("files", ("profile.txt", content, "text/plain"))],
    )
    assert process_response.status_code == 200
    assert process_response.json()["documents"][0]["status"] == "Ready"

    monkeypatch.setattr(
        chat,
        "call_ollama",
        lambda _prompt: "Hoa earned a Master of Science degree.",
    )

    chat_response = client.post(
        "/chat",
        json={"question": "What educational degree did Hoa earn?"},
    )

    assert chat_response.status_code == 200
    payload = chat_response.json()
    assert payload["answer"] == "Hoa earned a Master of Science degree."
    assert payload["sources"][0]["filename"] == "profile.txt"
    assert payload["sources"][0]["snippet"] == content.decode("utf-8")


def test_chat_is_scoped_to_session_id(monkeypatch) -> None:
    alpha_content = b"Alpha document discusses contract renewal dates."
    beta_content = b"Beta document discusses salary details."
    process_alpha = client.post(
        "/documents/process",
        data={"session_id": "alpha"},
        files=[("files", ("alpha.txt", alpha_content, "text/plain"))],
    )
    process_beta = client.post(
        "/documents/process",
        data={"session_id": "beta"},
        files=[("files", ("beta.txt", beta_content, "text/plain"))],
    )
    assert process_alpha.status_code == 200
    assert process_beta.status_code == 200

    monkeypatch.setattr(
        chat,
        "call_ollama",
        lambda _prompt: "Alpha document discusses contract renewal dates.",
    )

    alpha_response = client.post(
        "/chat",
        json={"session_id": "alpha", "question": "contract renewal"},
    )
    beta_response = client.post(
        "/chat",
        json={"session_id": "beta", "question": "contract renewal"},
    )

    assert alpha_response.status_code == 200
    assert alpha_response.json()["sources"][0]["filename"] == "alpha.txt"
    assert beta_response.status_code == 200
    assert beta_response.json()["sources"] == []


def test_chat_returns_503_when_ollama_is_unavailable(monkeypatch) -> None:
    chunks = process_document("guide.txt", b"This document describes citations.")
    document_index.replace_chunks(chunks)

    def unavailable_ollama(_prompt: str) -> str:
        raise chat.OllamaError("Ollama is not available.")

    monkeypatch.setattr(chat, "call_ollama", unavailable_ollama)

    response = client.post("/chat", json={"question": "citations"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Ollama is not available."}


def test_call_ollama_posts_prompt_to_configured_model(monkeypatch) -> None:
    requests = []

    def fake_urlopen(request, timeout: int) -> FakeOllamaResponse:
        requests.append((request, timeout))
        return FakeOllamaResponse(b'{"response": "Grounded answer."}')

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.example.test")
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    monkeypatch.setattr(chat, "urlopen", fake_urlopen)

    answer = chat.call_ollama("Use this context only.")

    request, timeout = requests[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert answer == "Grounded answer."
    assert request.full_url == "http://ollama.example.test/api/generate"
    assert timeout == 120
    assert payload == {
        "model": "test-model",
        "prompt": "Use this context only.",
        "stream": False,
        "options": {"temperature": 0},
    }


def test_call_ollama_reports_unavailable_server(monkeypatch) -> None:
    def fake_urlopen(_request, timeout: int) -> FakeOllamaResponse:
        raise URLError("connection refused")

    monkeypatch.setattr(chat, "urlopen", fake_urlopen)

    try:
        chat.call_ollama("prompt")
    except chat.OllamaError as exc:
        assert str(exc) == (
            "Ollama is not available. Start Ollama and pull the configured model."
        )
    else:
        raise AssertionError("Expected OllamaError")


def test_call_ollama_includes_ollama_http_error_detail(monkeypatch) -> None:
    def fake_urlopen(_request, timeout: int) -> FakeOllamaResponse:
        assert timeout == 120
        raise HTTPError(
            url="http://localhost:11434/api/generate",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=BytesIO(b'{"error": "model llama3.2 not found"}'),
        )

    monkeypatch.setattr(chat, "urlopen", fake_urlopen)

    try:
        chat.call_ollama("prompt")
    except chat.OllamaError as exc:
        assert str(exc) == "Ollama returned HTTP 404: model llama3.2 not found"
    else:
        raise AssertionError("Expected OllamaError")
