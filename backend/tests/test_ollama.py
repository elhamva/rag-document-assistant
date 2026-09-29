import httpx
import pytest

from backend.app.ollama import EMBED_BATCH_SIZE, OllamaClient, OllamaError


def fake_post(calls: list, response: httpx.Response):
    def post(url: str, json: dict, timeout: float) -> httpx.Response:
        calls.append((url, json))
        return response

    return post


def test_generate_posts_prompt_to_configured_model(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(httpx, "post", fake_post(calls, httpx.Response(200, json={"response": "Hi"})))

    answer = OllamaClient(base_url="http://ollama.test", model="test-model").generate("prompt")

    assert answer == "Hi"
    assert calls == [
        (
            "http://ollama.test/api/generate",
            {"model": "test-model", "prompt": "prompt", "stream": False, "options": {"temperature": 0}},
        )
    ]


def test_generate_passes_json_schema_as_format(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(httpx, "post", fake_post(calls, httpx.Response(200, json={"response": "{}"})))
    schema = {"type": "object"}

    OllamaClient().generate("prompt", json_schema=schema)

    assert calls[0][1]["format"] == schema


def test_nomic_embeddings_use_task_prefixes_and_batches(monkeypatch) -> None:
    calls = []

    def post(url: str, json: dict, timeout: float) -> httpx.Response:
        calls.append(json["input"])
        return httpx.Response(200, json={"embeddings": [[1.0]] * len(json["input"])})

    monkeypatch.setattr(httpx, "post", post)
    client = OllamaClient(embed_model="nomic-embed-text")

    vectors = client.embed_documents(["text"] * (EMBED_BATCH_SIZE + 1))
    client.embed_query("question")

    assert len(vectors) == EMBED_BATCH_SIZE + 1
    assert [len(batch) for batch in calls] == [EMBED_BATCH_SIZE, 1, 1]
    assert calls[0][0] == "search_document: text"
    assert calls[2] == ["search_query: question"]


def test_unreachable_server_raises_clear_error(monkeypatch) -> None:
    def post(*_args, **_kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", post)

    with pytest.raises(OllamaError, match="Cannot reach Ollama"):
        OllamaClient().generate("prompt")


def test_http_error_includes_ollama_detail(monkeypatch) -> None:
    response = httpx.Response(404, json={"error": 'model "llama3.2:3b" not found'})
    monkeypatch.setattr(httpx, "post", fake_post([], response))

    with pytest.raises(OllamaError, match='HTTP 404: model "llama3.2:3b" not found'):
        OllamaClient().generate("prompt")
