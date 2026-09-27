import json
from io import BytesIO
from urllib.error import HTTPError, URLError

import streamlit as st

from frontend import app as frontend_app


class FakeHealthResponse:
    status = 200

    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "FakeHealthResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class FakeUploadedFile:
    def __init__(self, name: str, content: bytes, file_type: str) -> None:
        self.name = name
        self.size = len(content)
        self.type = file_type
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def test_check_backend_health_calls_existing_health_endpoint(monkeypatch) -> None:
    requests = []

    def fake_urlopen(request, timeout: float) -> FakeHealthResponse:
        requests.append((request.full_url, request.get_header("Accept"), timeout))
        return FakeHealthResponse(
            b'{"status": "ok", "service": "document-chat-api"}'
        )

    monkeypatch.setattr(frontend_app, "urlopen", fake_urlopen)

    status = frontend_app.check_backend_health(
        "http://api.example.test/",
        timeout_seconds=1.5,
    )

    assert status == frontend_app.BackendStatus(
        available=True,
        detail="Connected to document-chat-api.",
    )
    assert requests == [
        ("http://api.example.test/health", "application/json", 1.5),
    ]


def test_check_backend_health_reports_unavailable_backend(monkeypatch) -> None:
    def fake_urlopen(_request, timeout: float) -> FakeHealthResponse:
        assert timeout == 2.0
        raise URLError("connection refused")

    monkeypatch.setattr(frontend_app, "urlopen", fake_urlopen)

    status = frontend_app.check_backend_health("http://api.example.test")

    assert status.available is False
    assert "Health check failed" in status.detail


def test_send_documents_to_backend_posts_uploaded_files(monkeypatch) -> None:
    st.session_state.clear()
    st.session_state.session_id = "session-test"
    uploaded_file = FakeUploadedFile("guide.txt", b"hello world", "text/plain")
    requests = []

    def fake_urlopen(request, timeout: float) -> FakeHealthResponse:
        requests.append((request, timeout))
        return FakeHealthResponse(
            (
                b'{"documents": [{"id": "guide.txt:11:text/plain", '
                b'"filename": "guide.txt", "status": "Ready", '
                b'"chunk_count": 1, "error": null}]}'
            )
        )

    monkeypatch.setattr(frontend_app, "urlopen", fake_urlopen)

    results = frontend_app.send_documents_to_backend(
        [uploaded_file],
        base_url="http://api.example.test",
        timeout_seconds=3.0,
    )

    request, timeout = requests[0]
    assert request.full_url == "http://api.example.test/documents/process"
    assert request.get_header("Content-type").startswith("multipart/form-data")
    assert b'name="session_id"' in request.data
    assert b"session-test" in request.data
    assert b'name="files"; filename="guide.txt"' in request.data
    assert b"hello world" in request.data
    assert timeout == 3.0
    assert results == [
        frontend_app.DocumentProcessingResult(
            id="guide.txt:11:text/plain",
            filename="guide.txt",
            status="Ready",
            chunk_count=1,
        )
    ]


def test_process_uploaded_documents_marks_backend_results_ready(monkeypatch) -> None:
    st.session_state.clear()
    uploaded_file = FakeUploadedFile("guide.txt", b"hello world", "text/plain")
    st.session_state.uploaded_documents = [
        {"id": "guide.txt:11:text/plain", "filename": "guide.txt", "status": "Waiting"}
    ]
    st.session_state.documents_processed = False
    st.session_state.ready_document_count = 0

    monkeypatch.setattr(
        frontend_app,
        "send_documents_to_backend",
        lambda _uploaded_files: [
            frontend_app.DocumentProcessingResult(
                id="guide.txt:11:text/plain",
                filename="guide.txt",
                status="Ready",
                chunk_count=1,
            )
        ],
    )

    status = frontend_app.process_uploaded_documents(
        [uploaded_file],
        health_checker=lambda: frontend_app.BackendStatus(
            available=True,
            detail="Connected to document-chat-api.",
        ),
    )

    assert status.ok is True
    assert st.session_state.uploaded_documents == [
        {
            "id": "guide.txt:11:text/plain",
            "filename": "guide.txt",
            "status": "Ready",
            "chunk_count": 1,
            "error": None,
        }
    ]
    assert st.session_state.documents_processed is True
    assert st.session_state.ready_document_count == 1


def test_process_uploaded_documents_keeps_waiting_when_backend_is_unavailable() -> None:
    st.session_state.clear()
    uploaded_file = FakeUploadedFile("guide.txt", b"hello world", "text/plain")
    st.session_state.uploaded_documents = [
        {"id": "guide.txt:11:text/plain", "filename": "guide.txt", "status": "Waiting"}
    ]
    st.session_state.documents_processed = False
    st.session_state.ready_document_count = 0

    status = frontend_app.process_uploaded_documents(
        [uploaded_file],
        health_checker=lambda: frontend_app.BackendStatus(
            available=False,
            detail="Health check failed: connection refused.",
        ),
    )

    assert status.ok is False
    assert st.session_state.uploaded_documents == [
        {"id": "guide.txt:11:text/plain", "filename": "guide.txt", "status": "Waiting"}
    ]
    assert st.session_state.documents_processed is False
    assert st.session_state.ready_document_count == 0


def test_send_chat_to_backend_posts_question(monkeypatch) -> None:
    st.session_state.clear()
    st.session_state.session_id = "session-test"
    requests = []

    def fake_urlopen(request, timeout: float) -> FakeHealthResponse:
        requests.append((request, timeout))
        return FakeHealthResponse(
            (
                b'{"answer": "Citation metadata is kept.", '
                b'"sources": [{"chunk_id": "chunk-1", "filename": "guide.txt", '
                b'"page": null, "snippet": "citation metadata", "score": 1.0}]}'
            )
        )

    monkeypatch.setattr(frontend_app, "urlopen", fake_urlopen)

    answer = frontend_app.send_chat_to_backend(
        "What metadata is kept?",
        history=[{"role": "user", "content": "Tell me about citations"}],
        base_url="http://api.example.test",
        timeout_seconds=4.0,
    )

    request, timeout = requests[0]
    assert request.full_url == "http://api.example.test/chat"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data.decode("utf-8")) == {
        "question": "What metadata is kept?",
        "session_id": "session-test",
        "history": [{"role": "user", "content": "Tell me about citations"}],
    }
    assert timeout == 4.0
    assert answer == frontend_app.ChatAnswer(
        content="Citation metadata is kept.",
        sources=[
            {
                "chunk_id": "chunk-1",
                "filename": "guide.txt",
                "page": None,
                "snippet": "citation metadata",
                "score": 1.0,
            }
        ],
    )


def test_send_chat_to_backend_includes_backend_error_detail(monkeypatch) -> None:
    st.session_state.clear()
    st.session_state.session_id = "session-test"

    def fake_urlopen(_request, timeout: float) -> FakeHealthResponse:
        assert timeout == 30.0
        raise HTTPError(
            url="http://api.example.test/chat",
            code=503,
            msg="Service Unavailable",
            hdrs={},
            fp=BytesIO(
                b'{"detail": "Ollama is not available. Start Ollama and pull the configured model."}'
            ),
        )

    monkeypatch.setattr(frontend_app, "urlopen", fake_urlopen)

    try:
        frontend_app.send_chat_to_backend(
            "What metadata is kept?",
            base_url="http://api.example.test",
        )
    except frontend_app.BackendRequestError as exc:
        assert str(exc) == (
            "Chat returned HTTP 503: Ollama is not available. "
            "Start Ollama and pull the configured model."
        )
    else:
        raise AssertionError("Expected BackendRequestError")


def test_add_chat_response_preserves_history_with_backend_answer(monkeypatch) -> None:
    st.session_state.clear()
    calls = []
    st.session_state.chat_messages = [
        {"role": "user", "content": "Tell me about citations"},
        {"role": "assistant", "content": "Citation metadata is used.", "sources": []},
    ]

    monkeypatch.setattr(
        frontend_app,
        "send_chat_to_backend",
        lambda question, history: calls.append((question, history))
        or frontend_app.ChatAnswer(
            content="Citation metadata is kept.",
            sources=[
                {
                    "chunk_id": "chunk-1",
                    "filename": "guide.txt",
                    "page": None,
                    "snippet": "citation metadata",
                    "score": 1.0,
                }
            ],
        ),
    )

    frontend_app.add_chat_response("What metadata is kept?")

    assert st.session_state.chat_messages == [
        {"role": "user", "content": "Tell me about citations"},
        {"role": "assistant", "content": "Citation metadata is used.", "sources": []},
        {"role": "user", "content": "What metadata is kept?"},
        {
            "role": "assistant",
            "content": "Citation metadata is kept.",
            "sources": [
                {
                    "chunk_id": "chunk-1",
                    "filename": "guide.txt",
                    "page": None,
                    "snippet": "citation metadata",
                    "score": 1.0,
                }
            ],
        },
    ]
    assert calls == [
        (
            "What metadata is kept?",
            [
                {"role": "user", "content": "Tell me about citations"},
                {"role": "assistant", "content": "Citation metadata is used."},
            ],
        )
    ]
