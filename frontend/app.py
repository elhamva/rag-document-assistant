from __future__ import annotations

import json
import os
from dataclasses import dataclass
from html import escape
from typing import Any, Callable, Sequence
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

import streamlit as st


DocumentRecord = dict[str, Any]
SourceRecord = dict[str, Any]

BACKEND_URL_ENV_VAR = "DOCUMENT_CHAT_API_URL"
DEFAULT_BACKEND_URL = "http://localhost:8000"
SUPPORTED_EXTENSIONS = {"pdf", "txt"}


@dataclass(frozen=True)
class BackendStatus:
    available: bool
    detail: str


@dataclass(frozen=True)
class DocumentProcessingResult:
    id: str
    filename: str
    status: str
    chunk_count: int
    error: str | None = None


@dataclass(frozen=True)
class DocumentProcessingStatus:
    ok: bool
    detail: str


@dataclass(frozen=True)
class ChatAnswer:
    content: str
    sources: list[SourceRecord]


class BackendRequestError(RuntimeError):
    pass


def initialize_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex

    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = []

    if "ready_document_count" not in st.session_state:
        st.session_state.ready_document_count = 0

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = bool(st.session_state.uploaded_documents)


def get_backend_base_url() -> str:
    return os.getenv(BACKEND_URL_ENV_VAR, DEFAULT_BACKEND_URL).rstrip("/")


def check_backend_health(
    base_url: str | None = None,
    timeout_seconds: float = 2.0,
) -> BackendStatus:
    api_base_url = (base_url or get_backend_base_url()).rstrip("/")
    request = Request(
        f"{api_base_url}/health",
        headers={"Accept": "application/json"},
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status_code = getattr(response, "status", None)
            if status_code is None:
                status_code = response.getcode()

            if status_code != 200:
                return BackendStatus(
                    available=False,
                    detail=f"Health check returned HTTP {status_code}.",
                )

            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return BackendStatus(
            available=False,
            detail=f"Health check returned HTTP {exc.code}.",
        )
    except (OSError, json.JSONDecodeError) as exc:
        return BackendStatus(
            available=False,
            detail=f"Health check failed: {exc}.",
        )

    if not isinstance(payload, dict):
        return BackendStatus(
            available=False,
            detail="Health check response was invalid.",
        )

    service = payload.get("service")
    if payload.get("status") == "ok" and isinstance(service, str) and service:
        return BackendStatus(
            available=True,
            detail=f"Connected to {service}.",
        )

    return BackendStatus(
        available=False,
        detail="Health check response was invalid.",
    )


def render_documents_tab() -> None:
    st.markdown("#### Documents")
    st.caption("Supported formats: PDF and TXT.")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=sorted(SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="document_uploader",
    )
    sync_uploaded_documents(uploaded_files)

    action_cols = st.columns([3.2, 1.15])
    with action_cols[1]:
        process_clicked = st.button(
            "Process documents",
            type="primary",
            disabled=not st.session_state.uploaded_documents,
            use_container_width=True,
        )

    if process_clicked:
        processing_status = process_uploaded_documents(uploaded_files)
        if processing_status.ok:
            st.toast("Documents processed by backend.")
        else:
            st.error(processing_status.detail)

    render_processing_status()


def sync_uploaded_documents(uploaded_files: Sequence[Any] | None) -> None:
    uploaded_files = uploaded_files or []
    existing_documents = {
        document["id"]: document for document in st.session_state.uploaded_documents
    }

    next_documents: list[DocumentRecord] = []
    for uploaded_file in uploaded_files:
        document_id = get_document_id(uploaded_file)
        existing_document = existing_documents.get(document_id)

        if existing_document:
            document = existing_document.copy()
            document.update(build_document_record(uploaded_file, document["status"]))
        else:
            document = build_document_record(uploaded_file, "Waiting")

        next_documents.append(document)

    st.session_state.uploaded_documents = next_documents
    st.session_state.documents_processed = bool(next_documents) and all(
        document["status"] in {"Ready", "Failed"} for document in next_documents
    )
    update_ready_document_count()


def build_document_record(uploaded_file: Any, status: str) -> DocumentRecord:
    return {
        "id": get_document_id(uploaded_file),
        "filename": uploaded_file.name,
        "status": status,
    }


def get_document_id(uploaded_file: Any) -> str:
    file_type = getattr(uploaded_file, "type", "")
    return f"{uploaded_file.name}:{uploaded_file.size}:{file_type}"


def process_uploaded_documents(
    uploaded_files: Sequence[Any] | None,
    health_checker: Callable[[], BackendStatus] = check_backend_health,
) -> DocumentProcessingStatus:
    uploaded_files = uploaded_files or []
    backend_status = health_checker()

    if not backend_status.available:
        return DocumentProcessingStatus(ok=False, detail=backend_status.detail)

    try:
        processing_results = send_documents_to_backend(uploaded_files)
    except BackendRequestError as exc:
        return DocumentProcessingStatus(ok=False, detail=str(exc))

    apply_document_processing_results(processing_results)
    failed_count = sum(1 for result in processing_results if result.status == "Failed")

    if failed_count:
        return DocumentProcessingStatus(
            ok=False,
            detail=f"{failed_count} {pluralize('document', failed_count)} failed.",
        )

    return DocumentProcessingStatus(ok=True, detail="Documents processed.")


def send_documents_to_backend(
    uploaded_files: Sequence[Any],
    base_url: str | None = None,
    timeout_seconds: float = 30.0,
) -> list[DocumentProcessingResult]:
    api_base_url = (base_url or get_backend_base_url()).rstrip("/")
    body, content_type = build_multipart_body(uploaded_files)
    request = Request(
        f"{api_base_url}/documents/process",
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": content_type,
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status_code = getattr(response, "status", None)
            if status_code is None:
                status_code = response.getcode()

            if status_code != 200:
                raise BackendRequestError(
                    f"Document processing returned HTTP {status_code}."
                )

            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise BackendRequestError(
            f"Document processing returned HTTP {exc.code}."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendRequestError(f"Document processing failed: {exc}.") from exc

    return parse_processing_response(payload)


def build_multipart_body(uploaded_files: Sequence[Any]) -> tuple[bytes, str]:
    boundary = f"document-chat-{uuid4().hex}"
    body_parts: list[bytes] = [
        f"--{boundary}\r\n".encode("utf-8"),
        b'Content-Disposition: form-data; name="session_id"\r\n\r\n',
        st.session_state.session_id.encode("utf-8"),
        b"\r\n",
    ]

    for uploaded_file in uploaded_files:
        filename = quote_multipart_value(uploaded_file.name)
        content_type = getattr(uploaded_file, "type", "") or "application/octet-stream"
        content = uploaded_file.getvalue()

        body_parts.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    'Content-Disposition: form-data; name="files"; '
                    f'filename="{filename}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
                content,
                b"\r\n",
            ]
        )

    body_parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(body_parts), f"multipart/form-data; boundary={boundary}"


def quote_multipart_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def parse_processing_response(payload: Any) -> list[DocumentProcessingResult]:
    if not isinstance(payload, dict) or not isinstance(payload.get("documents"), list):
        raise BackendRequestError("Document processing response was invalid.")

    results: list[DocumentProcessingResult] = []
    for document in payload["documents"]:
        if not isinstance(document, dict):
            raise BackendRequestError("Document processing response was invalid.")

        document_id = document.get("id")
        filename = document.get("filename")
        status = document.get("status")
        chunk_count = document.get("chunk_count")
        error = document.get("error")

        if not isinstance(document_id, str) or not isinstance(filename, str):
            raise BackendRequestError("Document processing response was invalid.")

        if status not in {"Ready", "Failed"} or not isinstance(chunk_count, int):
            raise BackendRequestError("Document processing response was invalid.")

        results.append(
            DocumentProcessingResult(
                id=document_id,
                filename=filename,
                status=status,
                chunk_count=chunk_count,
                error=error if isinstance(error, str) else None,
            )
        )

    return results


def apply_document_processing_results(
    processing_results: list[DocumentProcessingResult],
) -> None:
    results_by_id = {result.id: result for result in processing_results}
    processed_documents: list[DocumentRecord] = []

    for document in st.session_state.uploaded_documents:
        processed_document = document.copy()
        result = results_by_id.get(document["id"])

        if result is None:
            processed_document["status"] = "Failed"
        else:
            processed_document["status"] = result.status
            processed_document["chunk_count"] = result.chunk_count
            processed_document["error"] = result.error

        processed_documents.append(processed_document)

    st.session_state.uploaded_documents = processed_documents
    st.session_state.documents_processed = bool(processed_documents)
    update_ready_document_count()


def render_processing_status() -> None:
    if not st.session_state.documents_processed:
        return

    failed_count = sum(
        1
        for document in st.session_state.uploaded_documents
        if document["status"] == "Failed"
    )

    if failed_count:
        st.error(f"{failed_count} {pluralize('document', failed_count)} failed")
        return

    ready_count = st.session_state.ready_document_count
    st.success(f"{ready_count} {pluralize('document', ready_count)} ready")


def render_chat_tab() -> None:
    ready_count = update_ready_document_count()
    st.markdown("#### Chat")
    st.caption(f"{ready_count} {pluralize('document', ready_count)} ready")

    if st.session_state.chat_messages:
        render_chat_history()
    elif ready_count > 0:
        st.caption("Ask a question about your uploaded document.")

    prompt = st.chat_input(
        "Ask a question",
        disabled=ready_count == 0,
    )

    if prompt and ready_count > 0:
        add_chat_response(prompt)
        st.rerun()


def render_chat_history() -> None:
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])


def render_sources(sources: list[SourceRecord]) -> None:
    with st.expander("Sources", expanded=False):
        for source in sources:
            page = source.get("page")
            page_label = f"Page {page}" if isinstance(page, int) else "Page unavailable"
            st.markdown(
                f"""
                <div class="source-row">
                    <div class="source-heading">
                        <span>{escape(source["filename"])}</span>
                        <span>{escape(page_label)}</span>
                    </div>
                    <div class="source-snippet">{escape(source["snippet"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def add_chat_response(prompt: str) -> None:
    history = build_chat_history_payload(st.session_state.chat_messages)
    st.session_state.chat_messages.append({"role": "user", "content": prompt})

    try:
        answer = send_chat_to_backend(prompt, history=history)
    except BackendRequestError as exc:
        answer = ChatAnswer(content=str(exc), sources=[])

    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": answer.content,
            "sources": answer.sources,
        }
    )


def send_chat_to_backend(
    question: str,
    history: list[dict[str, str]] | None = None,
    base_url: str | None = None,
    timeout_seconds: float = 30.0,
) -> ChatAnswer:
    api_base_url = (base_url or get_backend_base_url()).rstrip("/")
    request = Request(
        f"{api_base_url}/chat",
        data=json.dumps(
            {
                "question": question,
                "session_id": st.session_state.session_id,
                "history": history or [],
            }
        ).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status_code = getattr(response, "status", None)
            if status_code is None:
                status_code = response.getcode()

            if status_code != 200:
                raise BackendRequestError(f"Chat returned HTTP {status_code}.")

            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise BackendRequestError(format_http_error("Chat", exc)) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendRequestError(f"Chat failed: {exc}.") from exc

    return parse_chat_response(payload)


def build_chat_history_payload(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    history = []

    for message in messages[-6:]:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            history.append({"role": role, "content": content})

    return history


def format_http_error(action: str, exc: HTTPError) -> str:
    try:
        payload = json.loads(exc.read().decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        payload = None

    if isinstance(payload, dict) and isinstance(payload.get("detail"), str):
        return f"{action} returned HTTP {exc.code}: {payload['detail']}"

    return f"{action} returned HTTP {exc.code}."


def parse_chat_response(payload: Any) -> ChatAnswer:
    if not isinstance(payload, dict):
        raise BackendRequestError("Chat response was invalid.")

    answer = payload.get("answer")
    sources = payload.get("sources")
    if not isinstance(answer, str) or not isinstance(sources, list):
        raise BackendRequestError("Chat response was invalid.")

    parsed_sources: list[SourceRecord] = []
    for source in sources:
        if not isinstance(source, dict):
            raise BackendRequestError("Chat response was invalid.")

        filename = source.get("filename")
        page = source.get("page")
        snippet = source.get("snippet")
        score = source.get("score")
        chunk_id = source.get("chunk_id")

        if not isinstance(filename, str) or not isinstance(snippet, str):
            raise BackendRequestError("Chat response was invalid.")

        if page is not None and not isinstance(page, int):
            raise BackendRequestError("Chat response was invalid.")

        if not isinstance(score, (int, float)) or not isinstance(chunk_id, str):
            raise BackendRequestError("Chat response was invalid.")

        parsed_sources.append(
            {
                "chunk_id": chunk_id,
                "filename": filename,
                "page": page,
                "snippet": snippet,
                "score": float(score),
            }
        )

    return ChatAnswer(content=answer, sources=parsed_sources)


def update_ready_document_count() -> int:
    ready_count = sum(
        1
        for document in st.session_state.uploaded_documents
        if document["status"] == "Ready"
    )
    st.session_state.ready_document_count = ready_count
    return ready_count


def pluralize(noun: str, count: int) -> str:
    return noun if count == 1 else f"{noun}s"


def apply_page_styles() -> None:
    st.markdown(
        """
        <style>
            [data-testid="stAppViewContainer"] {
                background: #07111f;
            }

            [data-testid="stHeader"] {
                background: rgba(7, 17, 31, 0.92);
            }

            .block-container {
                max-width: 860px;
                padding-top: 2rem;
                padding-bottom: 2rem;
            }

            h4, h5 {
                letter-spacing: 0;
            }

            [data-baseweb="tab-list"] {
                gap: 0.5rem;
                margin-bottom: 0.9rem;
            }

            [data-testid="stVerticalBlockBorderWrapper"] {
                background: #0b1624;
                border: 1px solid #25364d;
                border-radius: 10px;
                min-height: 650px;
            }

            [data-testid="stVerticalBlockBorderWrapper"] > div {
                min-height: 620px;
                overflow: visible;
                padding: 1rem 1.1rem 1.1rem;
            }

            button[data-baseweb="tab"] {
                background: transparent;
                border-radius: 8px 8px 0 0;
                color: #c8d3e2;
                padding-left: 0.85rem;
                padding-right: 0.85rem;
            }

            button[data-baseweb="tab"][aria-selected="true"] {
                background: #101d30;
                color: #b9c7ff;
            }

            [data-testid="stFileUploader"] {
                margin: 0.6rem 0 0.9rem;
            }

            [data-testid="stFileUploaderDropzone"] {
                align-items: center;
                background: #101d30;
                border: 1px dashed #48688f;
                border-radius: 12px;
                display: flex;
                flex-direction: column;
                gap: 0.75rem;
                justify-content: center;
                min-height: 15rem;
                text-align: center;
            }

            [data-testid="stFileUploaderDropzoneInstructions"] {
                align-items: center;
                display: flex;
                flex-direction: column;
                margin: 0;
                text-align: center;
            }

            [data-testid="stFileUploaderDropzone"] > div {
                align-items: center;
                display: flex;
                flex-direction: column;
                gap: 0.55rem;
                justify-content: center;
                text-align: center;
                width: 100%;
            }

            [data-testid="stFileUploaderDropzone"] button {
                margin: 0.35rem auto 0;
            }

            [data-testid="stButton"] > button {
                background: #102a46;
                border: 1px solid #5579a6;
                border-radius: 12px;
                color: #eef5ff;
                min-height: 2.7rem;
            }

            [data-testid="stButton"] > button:hover:not(:disabled) {
                background: #17395f;
                border-color: #86a9d6;
                color: #ffffff;
            }

            [data-testid="stButton"] > button:disabled,
            [data-testid="stButton"] > button:disabled:hover {
                background: #0d2138;
                border-color: #335578;
                border-radius: 12px;
                color: #899bb2;
                opacity: 1;
            }

            .source-row {
                border-bottom: 1px solid #273142;
                padding: 0.75rem 0;
            }

            .source-row:last-child {
                border-bottom: none;
            }

            .source-heading {
                color: #f3f4f6;
                display: flex;
                font-size: 0.9rem;
                font-weight: 600;
                justify-content: space-between;
                gap: 1rem;
                margin-bottom: 0.35rem;
            }

            .source-snippet {
                color: #cbd5e1;
                font-size: 0.9rem;
                line-height: 1.45;
            }

            .source-score {
                color: #9ca3af;
                font-size: 0.82rem;
                margin-top: 0.3rem;
            }

            @media (max-width: 640px) {
                .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                    padding-top: 2rem;
                }

                [data-testid="stVerticalBlockBorderWrapper"] > div {
                    min-height: 520px;
                    padding: 1rem;
                }

                [data-testid="stFileUploaderDropzone"] {
                    min-height: 12rem;
                }

                .source-heading {
                    align-items: flex-start;
                    flex-direction: column;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Document Chat",
        page_icon="",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    initialize_state()
    apply_page_styles()

    documents_tab, chat_tab = st.tabs(["Documents", "Chat"])
    with documents_tab:
        with st.container(border=True):
            render_documents_tab()
    with chat_tab:
        with st.container(border=True):
            render_chat_tab()


if __name__ == "__main__":
    main()
