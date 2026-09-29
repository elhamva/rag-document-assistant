from __future__ import annotations

import os
import re
from html import escape
from typing import Any
from uuid import uuid4

import requests
import streamlit as st


BACKEND_URL = os.getenv("DOCUMENT_CHAT_API_URL", "http://localhost:8000").rstrip("/")
DEFAULT_ANSWER_MODEL = os.getenv("DOCUMENT_CHAT_MODEL", "llama3.2:3b")
DEFAULT_COMPARE_MODEL = os.getenv("DOCUMENT_CHAT_COMPARE_MODEL", "")
REQUEST_TIMEOUT_SECONDS = 300
HISTORY_MESSAGES = 6
HIGHLIGHT_STOP_WORDS = {
    "about",
    "above",
    "after",
    "answer",
    "because",
    "below",
    "does",
    "from",
    "have",
    "that",
    "their",
    "there",
    "these",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "within",
}


class BackendError(RuntimeError):
    pass


def post_to_backend(path: str, **kwargs: Any) -> dict:
    try:
        response = requests.post(f"{BACKEND_URL}{path}", timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
    except requests.RequestException as exc:
        raise BackendError(f"Backend is not reachable at {BACKEND_URL}.") from exc

    if not response.ok:
        try:
            detail = response.json()["detail"]
        except (ValueError, KeyError, TypeError):
            detail = response.text or response.reason
        raise BackendError(f"Backend error ({response.status_code}): {detail}")

    return response.json()


def process_documents(files: list, session_id: str) -> list[dict]:
    payload = [("files", (file.name, file.getvalue(), file.type or "application/octet-stream")) for file in files]
    return post_to_backend("/documents/process", files=payload, data={"session_id": session_id})["documents"]


def ask_question(question: str, history: list[dict], session_id: str, model: str | None = None) -> dict:
    payload = {"question": question, "history": history, "session_id": session_id}
    if model:
        payload["model"] = model
    return post_to_backend(
        "/chat",
        json=payload,
    )


def build_history(messages: list[dict]) -> list[dict]:
    turns = [
        {"role": message["role"], "content": message["content"]}
        for message in messages
        if not message.get("error")
    ]
    return turns[-HISTORY_MESSAGES:]


def file_signature(files: list) -> list[tuple[str, int]]:
    return [(file.name, file.size) for file in files]


def init_state() -> None:
    st.session_state.setdefault("session_id", uuid4().hex)
    st.session_state.setdefault("documents", [])
    st.session_state.setdefault("processed_files", [])
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("answer_model", DEFAULT_ANSWER_MODEL)
    st.session_state.setdefault("compare_enabled", False)
    st.session_state.setdefault("comparison_model", DEFAULT_COMPARE_MODEL)


def pluralize(noun: str, count: int) -> str:
    return noun if count == 1 else f"{noun}s"


def ready_count() -> int:
    return sum(1 for document in st.session_state.documents if document["status"] == "ready")


def selected_models() -> list[str]:
    primary = st.session_state.answer_model.strip() or DEFAULT_ANSWER_MODEL
    comparison = st.session_state.comparison_model.strip()
    models = [primary]
    if st.session_state.compare_enabled and comparison and comparison != primary:
        models.append(comparison)
    return models


def render_documents_tab() -> None:
    st.markdown("#### Documents")
    st.caption("Supported formats: PDF, TXT and Markdown.")

    files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    files = files or []
    # With no files left, the same request clears the backend index for this session.
    clearing = not files and bool(st.session_state.documents)
    action_cols = st.columns([3.2, 1.15])
    with action_cols[1]:
        process_clicked = st.button(
            "Clear documents" if clearing else "Process documents",
            type="primary",
            disabled=not files and not clearing,
            use_container_width=True,
        )

    if process_clicked:
        with st.spinner("Reading, chunking and embedding..."):
            try:
                st.session_state.documents = process_documents(files, st.session_state.session_id)
                st.session_state.processed_files = file_signature(files)
                st.toast("Documents cleared." if clearing else "Documents processed by backend.")
            except BackendError as exc:
                st.error(str(exc))

    render_processing_status(files)


def render_processing_status(files: list) -> None:
    if not st.session_state.documents:
        return

    if file_signature(files) != st.session_state.processed_files:
        st.info("The file list changed. Click Process documents to update the index.")

    for document in st.session_state.documents:
        if document["status"] == "failed":
            st.error(f"{document['filename']}: {document['error']}")

    count = ready_count()
    if count:
        st.success(f"{count} {pluralize('document', count)} ready")


def render_chat_tab() -> None:
    count = ready_count()
    st.markdown("#### Chat")
    st.caption(f"{count} {pluralize('document', count)} ready")

    with st.expander("Answer options", expanded=False):
        st.text_input("Primary model", key="answer_model")
        st.toggle("Compare two models", key="compare_enabled")
        if st.session_state.compare_enabled:
            st.text_input("Comparison model", key="comparison_model")

    conversation = st.container()
    with conversation:
        for message in st.session_state.messages:
            render_message(message)
        if not st.session_state.messages and count:
            st.caption("Ask a question about your uploaded documents.")

    question = st.chat_input("Ask a question", disabled=count == 0)
    if not question:
        return

    history = build_history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})

    with conversation:
        render_message(st.session_state.messages[-1])
        with st.chat_message("assistant"), st.spinner("Searching your documents..."):
            try:
                answers = []
                for model in selected_models():
                    answer = ask_question(question, history, st.session_state.session_id, model)
                    answers.append(
                        {
                            "model": model,
                            "answer": answer["answer"],
                            "sources": answer["sources"],
                        }
                    )
                reply = {
                    "role": "assistant",
                    "content": answers[0]["answer"],
                    "sources": answers[0]["sources"],
                    "model": answers[0]["model"],
                }
                if len(answers) > 1:
                    reply["comparisons"] = answers
            except BackendError as exc:
                reply = {"role": "assistant", "content": str(exc), "error": True}

    st.session_state.messages.append(reply)
    st.rerun()


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
            return

        if message.get("comparisons"):
            render_comparison(message["comparisons"])
            return

        render_answer(
            answer=message["content"],
            sources=message.get("sources", []),
            model=message.get("model"),
        )


def render_comparison(answers: list[dict]) -> None:
    tabs = st.tabs([answer["model"] for answer in answers])
    for tab, answer in zip(tabs, answers):
        with tab:
            render_answer(answer["answer"], answer["sources"], answer["model"])


def render_answer(answer: str, sources: list[dict], model: str | None = None) -> None:
    if model:
        st.caption(model)
    st.markdown(answer)
    if sources:
        render_sources(sources, answer)
        render_artifact(answer, sources, model)


def render_sources(sources: list[dict], answer: str) -> None:
    with st.expander("Highlighted sources", expanded=False):
        for source in sources:
            page = f"Page {source['page']}" if source.get("page") else "No page"
            st.markdown(
                f"""
                <div class="source-row">
                    <div class="source-heading">
                        <span>{escape(source["filename"])}</span>
                        <span>{page}</span>
                    </div>
                    <div class="source-snippet">{highlight_source(source["text"], answer)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_artifact(answer: str, sources: list[dict], model: str | None) -> None:
    artifact = build_answer_artifact(answer, sources, model)
    with st.expander("Answer artifact", expanded=False):
        st.markdown(artifact)
        st.download_button(
            "Download markdown",
            data=artifact,
            file_name="document-chat-answer.md",
            mime="text/markdown",
            use_container_width=True,
        )


def build_answer_artifact(answer: str, sources: list[dict], model: str | None = None) -> str:
    lines = ["## Answer", "", answer.strip(), ""]
    if model:
        lines += [f"Model: `{model}`", ""]
    lines += ["## Sources"]
    for index, source in enumerate(sources, start=1):
        page = f", page {source['page']}" if source.get("page") else ""
        lines += [
            "",
            f"{index}. `{source['filename']}`{page}",
            "",
            f"> {source['text']}",
        ]
    return "\n".join(lines)


def highlight_source(text: str, answer: str) -> str:
    terms = citation_terms(answer)
    if not terms:
        return escape(text)

    html = []
    for part in re.split(r"(\W+)", text):
        normalized = part.lower().strip("_")
        if normalized in terms:
            html.append(f"<mark>{escape(part)}</mark>")
        else:
            html.append(escape(part))
    return "".join(html)


def citation_terms(answer: str) -> set[str]:
    terms = set()
    for token in re.findall(r"[A-Za-z0-9]+", answer.lower()):
        # Numbers are usually the fact that was asked for, so keep them even when short ("45", "1,500").
        if (len(token) >= 4 or any(char.isdigit() for char in token)) and token not in HIGHLIGHT_STOP_WORDS:
            terms.add(token)
    return terms


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

            .source-snippet mark {
                background: #f5c542;
                border-radius: 3px;
                color: #111827;
                padding: 0 0.14rem;
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
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    init_state()
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
