from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Sequence

import streamlit as st


DocumentRecord = dict[str, Any]
SourceRecord = dict[str, Any]

SUPPORTED_EXTENSIONS = {"pdf", "txt"}


def initialize_state() -> None:
    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = []

    if "ready_document_count" not in st.session_state:
        st.session_state.ready_document_count = 0

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = bool(st.session_state.uploaded_documents)


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
        simulate_document_processing()
        st.toast("Documents marked ready in mock mode.")

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


def simulate_document_processing() -> None:
    if not st.session_state.uploaded_documents:
        return

    processed_documents: list[DocumentRecord] = []

    for document in st.session_state.uploaded_documents:
        processed_document = document.copy()
        file_extension = Path(processed_document["filename"]).suffix.lower().lstrip(".")

        if file_extension not in SUPPORTED_EXTENSIONS:
            processed_document["status"] = "Failed"
        else:
            processed_document["status"] = "Ready"

        processed_documents.append(processed_document)

    st.session_state.uploaded_documents = processed_documents
    st.session_state.documents_processed = True
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
        add_mock_chat_response(prompt)
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
            st.markdown(
                f"""
                <div class="source-row">
                    <div class="source-heading">
                        <span>{escape(source["filename"])}</span>
                        <span>Page {source["page"]}</span>
                    </div>
                    <div class="source-snippet">{escape(source["snippet"])}</div>
                    <div class="source-score">Retrieval score: {source["score"]:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def add_mock_chat_response(prompt: str) -> None:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": (
                "This is a placeholder answer. The backend will later replace this response "
                "with retrieval results and a model-generated answer."
            ),
            "sources": build_mock_sources(),
        }
    )


def build_mock_sources() -> list[SourceRecord]:
    ready_documents = [
        document
        for document in st.session_state.uploaded_documents
        if document["status"] == "Ready"
    ]
    first_filename = ready_documents[0]["filename"] if ready_documents else "sample.pdf"
    second_filename = (
        ready_documents[1]["filename"]
        if len(ready_documents) > 1
        else first_filename
    )

    return [
        {
            "filename": first_filename,
            "page": 2,
            "snippet": "The document introduces the primary requirements and expected review flow.",
            "score": 0.91,
        },
        {
            "filename": second_filename,
            "page": 5,
            "snippet": "A later section lists unresolved items that should be checked before implementation.",
            "score": 0.84,
        },
    ]


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
