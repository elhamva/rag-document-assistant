import pytest
import requests

from frontend import app


class FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self.ok = status_code < 400
        self.reason = "Error"
        self.text = ""
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class FakeUploadedFile:
    def __init__(self, name: str, content: bytes, type: str = "text/plain") -> None:
        self.name = name
        self.size = len(content)
        self.type = type
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def test_process_documents_sends_files_and_session(monkeypatch) -> None:
    calls = []

    def fake_post(url: str, timeout: float, **kwargs) -> FakeResponse:
        calls.append((url, kwargs))
        return FakeResponse(200, {"documents": [{"filename": "a.txt", "status": "ready"}]})

    monkeypatch.setattr(requests, "post", fake_post)

    documents = app.process_documents([FakeUploadedFile("a.txt", b"hello")], session_id="abc")

    assert documents == [{"filename": "a.txt", "status": "ready"}]
    url, kwargs = calls[0]
    assert url.endswith("/documents/process")
    assert kwargs["files"] == [("files", ("a.txt", b"hello", "text/plain"))]
    assert kwargs["data"] == {"session_id": "abc"}


def test_ask_question_sends_history_and_session(monkeypatch) -> None:
    calls = []

    def fake_post(url: str, timeout: float, **kwargs) -> FakeResponse:
        calls.append(kwargs["json"])
        return FakeResponse(200, {"answer": "Five years.", "sources": []})

    monkeypatch.setattr(requests, "post", fake_post)
    history = [{"role": "user", "content": "Hi"}]

    answer = app.ask_question("How long?", history, session_id="abc")

    assert answer == {"answer": "Five years.", "sources": []}
    assert calls == [{"question": "How long?", "history": history, "session_id": "abc"}]


def test_ask_question_can_send_model(monkeypatch) -> None:
    calls = []

    def fake_post(url: str, timeout: float, **kwargs) -> FakeResponse:
        calls.append(kwargs["json"])
        return FakeResponse(200, {"answer": "Five years.", "sources": []})

    monkeypatch.setattr(requests, "post", fake_post)

    app.ask_question("How long?", [], session_id="abc", model="qwen2.5:14b")

    assert calls == [
        {
            "question": "How long?",
            "history": [],
            "session_id": "abc",
            "model": "qwen2.5:14b",
        }
    ]


def test_backend_error_detail_is_shown(monkeypatch) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: FakeResponse(503, {"detail": "Cannot reach Ollama."}),
    )

    with pytest.raises(app.BackendError, match="Backend error \\(503\\): Cannot reach Ollama."):
        app.ask_question("question", [], session_id="abc")


def test_unreachable_backend_gives_clear_error(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(app.BackendError, match="Backend is not reachable"):
        app.ask_question("question", [], session_id="abc")


def test_history_skips_errors_and_keeps_recent_turns() -> None:
    messages = [{"role": "user", "content": f"q{i}"} for i in range(10)]
    messages.append({"role": "assistant", "content": "Backend error", "error": True})

    history = app.build_history(messages)

    assert len(history) == app.HISTORY_MESSAGES
    assert history[-1] == {"role": "user", "content": "q9"}


def test_highlight_source_marks_answer_terms() -> None:
    html = app.highlight_source(
        "Shipping is free of charge above 1,500 euros.",
        "No, shipping is free of charge for orders above 1,500 euros.",
    )

    assert "<mark>Shipping</mark>" in html
    assert "<mark>free</mark>" in html
    assert "<mark>euros</mark>" in html


def test_answer_artifact_contains_answer_model_and_sources() -> None:
    artifact = app.build_answer_artifact(
        "Five years.",
        [{"filename": "handbook.md", "page": 2, "text": "Warranty: five years."}],
        model="llama3.2:3b",
    )

    assert "## Answer" in artifact
    assert "Model: `llama3.2:3b`" in artifact
    assert "`handbook.md`, page 2" in artifact
    assert "> Warranty: five years." in artifact
