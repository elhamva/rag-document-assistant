from urllib.error import URLError

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


def test_process_documents_marks_ready_only_after_backend_is_available() -> None:
    st.session_state.clear()
    st.session_state.uploaded_documents = [
        {"id": "guide.txt:12:text/plain", "filename": "guide.txt", "status": "Waiting"}
    ]
    st.session_state.documents_processed = False
    st.session_state.ready_document_count = 0

    status = frontend_app.process_documents_with_backend_health(
        lambda: frontend_app.BackendStatus(
            available=True,
            detail="Connected to document-chat-api.",
        )
    )

    assert status.available is True
    assert st.session_state.uploaded_documents == [
        {"id": "guide.txt:12:text/plain", "filename": "guide.txt", "status": "Ready"}
    ]
    assert st.session_state.documents_processed is True
    assert st.session_state.ready_document_count == 1


def test_process_documents_keeps_waiting_when_backend_is_unavailable() -> None:
    st.session_state.clear()
    st.session_state.uploaded_documents = [
        {"id": "guide.txt:12:text/plain", "filename": "guide.txt", "status": "Waiting"}
    ]
    st.session_state.documents_processed = False
    st.session_state.ready_document_count = 0

    status = frontend_app.process_documents_with_backend_health(
        lambda: frontend_app.BackendStatus(
            available=False,
            detail="Health check failed: connection refused.",
        )
    )

    assert status.available is False
    assert st.session_state.uploaded_documents == [
        {"id": "guide.txt:12:text/plain", "filename": "guide.txt", "status": "Waiting"}
    ]
    assert st.session_state.documents_processed is False
    assert st.session_state.ready_document_count == 0
