from backend.app.rag import NO_ANSWER


WARRANTY_TEXT = b"The warranty for LED luminaires lasts five years from delivery."


def upload(client, *files, session_id: str = "default"):
    return client.post(
        "/documents/process",
        data={"session_id": session_id},
        files=[("files", (name, content, "text/plain")) for name, content in files],
    )


def test_process_reports_each_document(client) -> None:
    response = upload(client, ("warranty.txt", WARRANTY_TEXT), ("image.png", b"binary"))

    assert response.status_code == 200
    assert response.json()["documents"] == [
        {"filename": "warranty.txt", "status": "ready", "chunk_count": 1, "error": None},
        {
            "filename": "image.png",
            "status": "failed",
            "chunk_count": 0,
            "error": "Unsupported file type. Upload PDF, TXT or MD files.",
        },
    ]


def test_reprocessing_an_unchanged_file_does_not_embed_it_again(client, embedder) -> None:
    upload(client, ("warranty.txt", WARRANTY_TEXT))
    upload(client, ("warranty.txt", WARRANTY_TEXT), ("fruit.txt", b"Bananas are fruit."))

    assert embedder.embedded_texts == [WARRANTY_TEXT.decode(), "Bananas are fruit."]


def test_removed_files_leave_the_index(client, index) -> None:
    upload(client, ("warranty.txt", WARRANTY_TEXT), ("fruit.txt", b"Bananas are fruit."))
    upload(client, ("fruit.txt", b"Bananas are fruit."))

    assert [chunk.filename for chunk in index.chunks("default")] == ["fruit.txt"]


def test_process_returns_503_when_embeddings_are_unavailable(client, embedder) -> None:
    embedder.available = False

    response = upload(client, ("warranty.txt", WARRANTY_TEXT))

    assert response.status_code == 503
    assert "model not found" in response.json()["detail"]


def test_chat_answers_from_retrieved_chunks(client, llm) -> None:
    upload(client, ("warranty.txt", WARRANTY_TEXT))
    llm.answer = "Five years."

    response = client.post("/chat", json={"question": "How long is the warranty?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Five years.",
        "sources": [{"filename": "warranty.txt", "page": None, "text": WARRANTY_TEXT.decode()}],
    }
    assert WARRANTY_TEXT.decode() in llm.prompts[0]
    assert "Question: How long is the warranty?" in llm.prompts[0]


def test_follow_up_question_uses_previous_question_for_retrieval(client, llm) -> None:
    upload(client, ("warranty.txt", WARRANTY_TEXT))

    response = client.post(
        "/chat",
        json={
            "question": "and how long is it?",
            "history": [
                {"role": "user", "content": "Is there a warranty?"},
                {"role": "assistant", "content": "Yes."},
            ],
        },
    )

    assert response.json()["sources"][0]["filename"] == "warranty.txt"
    assert "user: Is there a warranty?" in llm.prompts[0]


def test_chat_without_matching_context_does_not_call_the_model(client, llm) -> None:
    upload(client, ("warranty.txt", WARRANTY_TEXT))

    response = client.post("/chat", json={"question": "Who won the 2014 world cup?"})

    assert response.json() == {"answer": NO_ANSWER, "sources": []}
    assert llm.prompts == []


def test_model_refusal_hides_sources(client, llm) -> None:
    upload(client, ("warranty.txt", WARRANTY_TEXT))
    llm.answer = NO_ANSWER

    response = client.post("/chat", json={"question": "What does the warranty cost?"})

    assert response.json() == {"answer": NO_ANSWER, "sources": []}


def test_chat_is_scoped_to_session(client) -> None:
    upload(client, ("warranty.txt", WARRANTY_TEXT), session_id="alpha")

    response = client.post("/chat", json={"session_id": "beta", "question": "warranty"})

    assert response.json()["sources"] == []


def test_chat_returns_503_when_ollama_is_down(client, llm) -> None:
    upload(client, ("warranty.txt", WARRANTY_TEXT))
    llm.available = False

    response = client.post("/chat", json={"question": "warranty"})

    assert response.status_code == 503
    assert "Cannot reach Ollama" in response.json()["detail"]


def test_chat_rejects_blank_question(client) -> None:
    assert client.post("/chat", json={"question": "   "}).status_code == 422
