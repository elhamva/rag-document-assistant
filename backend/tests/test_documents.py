import json

from fastapi.testclient import TestClient

from backend.app import document_index as document_index_module
from backend.app.document_index import (
    DocumentIndex,
    EmbeddingError,
    EmbeddingProvider,
    IndexedChunk,
    OllamaEmbeddingProvider,
    Reranker,
    SearchResult,
    document_index,
    reciprocal_rank_fusion,
)
from backend.app.main import app


client = TestClient(app)


class FailingEmbeddingProvider(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("Embeddings unavailable.")


class SemanticTestEmbeddingProvider(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        text = text.lower()
        if any(term in text for term in ("automobile", "car", "vehicle")):
            return [1.0, 0.0]

        if any(term in text for term in ("banana", "fruit")):
            return [0.0, 1.0]

        return [0.0, 0.0]


class PassthroughReranker(Reranker):
    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        return candidates


class PreferredTextReranker(Reranker):
    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        reranked = [
            SearchResult(
                chunk=candidate.chunk,
                score=2.0 if "preferred" in candidate.chunk.text else 1.0,
            )
            for candidate in candidates
        ]
        reranked.sort(key=lambda result: result.score, reverse=True)
        return reranked


class FakeEmbeddingResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "FakeEmbeddingResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def setup_function() -> None:
    document_index._embedding_provider = FailingEmbeddingProvider()
    document_index._reranker = PassthroughReranker()
    document_index.clear()


def test_process_txt_document_indexes_chunks() -> None:
    content = (
        b"Retrieval augmented generation uses indexed document chunks. "
        b"Those chunks keep citation metadata for later answers."
    )

    response = client.post(
        "/documents/process",
        files=[("files", ("guide.txt", content, "text/plain"))],
    )

    assert response.status_code == 200
    assert response.json() == {
        "documents": [
            {
                "id": f"guide.txt:{len(content)}:text/plain",
                "filename": "guide.txt",
                "status": "Ready",
                "chunk_count": 1,
                "error": None,
            }
        ]
    }

    chunks = document_index.list_chunks()
    assert len(chunks) == 1
    assert chunks[0].filename == "guide.txt"
    assert chunks[0].page_number is None
    assert "indexed document chunks" in chunks[0].text
    assert document_index.search("citation metadata")[0].id == chunks[0].id
    assert document_index.search("what is the and of") == []


def test_process_pdf_document_indexes_page_metadata() -> None:
    content = build_test_pdf("RAG PDF citation text")

    response = client.post(
        "/documents/process",
        files=[("files", ("guide.pdf", content, "application/pdf"))],
    )

    assert response.status_code == 200
    document = response.json()["documents"][0]
    assert document["id"] == f"guide.pdf:{len(content)}:application/pdf"
    assert document["filename"] == "guide.pdf"
    assert document["status"] == "Ready"
    assert document["chunk_count"] == 1
    assert document["error"] is None

    chunks = document_index.list_chunks()
    assert len(chunks) == 1
    assert chunks[0].filename == "guide.pdf"
    assert chunks[0].page_number == 1
    assert "RAG PDF citation text" in chunks[0].text


def test_process_document_marks_unsupported_files_failed() -> None:
    response = client.post(
        "/documents/process",
        files=[("files", ("image.png", b"not a document", "image/png"))],
    )

    assert response.status_code == 200
    document = response.json()["documents"][0]
    assert document["filename"] == "image.png"
    assert document["status"] == "Failed"
    assert document["chunk_count"] == 0
    assert document["error"] == "Unsupported file type."
    assert document_index.list_chunks() == []


def test_process_document_marks_corrupt_pdf_failed() -> None:
    response = client.post(
        "/documents/process",
        files=[("files", ("broken.pdf", b"not a pdf", "application/pdf"))],
    )

    assert response.status_code == 200
    document = response.json()["documents"][0]
    assert document["filename"] == "broken.pdf"
    assert document["status"] == "Failed"
    assert document["chunk_count"] == 0
    assert document["error"] == "PDF text could not be extracted."
    assert document_index.list_chunks() == []


def test_semantic_search_ranks_related_chunk_without_exact_keyword() -> None:
    index = DocumentIndex(
        embedding_provider=SemanticTestEmbeddingProvider(),
        reranker=PassthroughReranker(),
    )
    vehicle_chunk = IndexedChunk(
        id="vehicle",
        filename="notes.txt",
        page_number=None,
        text="The car requires regular maintenance.",
    )
    fruit_chunk = IndexedChunk(
        id="fruit",
        filename="notes.txt",
        page_number=None,
        text="Bananas are yellow fruit.",
    )

    index.replace_chunks([fruit_chunk, vehicle_chunk])

    results = index.search_with_scores("automobile upkeep", limit=1)

    assert results[0].chunk.id == "vehicle"
    assert results[0].score > 0


def test_lexical_search_is_used_when_embeddings_are_unavailable() -> None:
    index = DocumentIndex(
        embedding_provider=FailingEmbeddingProvider(),
        reranker=PassthroughReranker(),
    )
    chunk = IndexedChunk(
        id="citation",
        filename="guide.txt",
        page_number=None,
        text="Citation metadata is stored with each chunk.",
    )

    index.replace_chunks([chunk])

    assert index.embedding_available is False
    assert index.search_with_scores("citation metadata", limit=1)[0].chunk.id == "citation"


def test_hybrid_search_runs_dense_and_lexical_retrieval(monkeypatch) -> None:
    index = DocumentIndex(
        embedding_provider=FailingEmbeddingProvider(),
        reranker=PassthroughReranker(),
    )
    dense_chunk = IndexedChunk(
        id="dense",
        filename="dense.txt",
        page_number=None,
        text="Dense match.",
    )
    lexical_chunk = IndexedChunk(
        id="lexical",
        filename="lexical.txt",
        page_number=None,
        text="Lexical match.",
    )
    calls = []

    def semantic_search(query: str, limit: int, session_id: str) -> list[SearchResult]:
        calls.append(("dense", query, limit, session_id))
        return [SearchResult(chunk=dense_chunk, score=0.9)]

    def lexical_search(query: str, limit: int, session_id: str) -> list[SearchResult]:
        calls.append(("lexical", query, limit, session_id))
        return [SearchResult(chunk=lexical_chunk, score=0.8)]

    monkeypatch.setattr(index, "_semantic_search", semantic_search)
    monkeypatch.setattr(index, "_lexical_search", lexical_search)

    results = index.search_with_scores("query", limit=2, session_id="session-a")

    assert [call[0] for call in calls] == ["dense", "lexical"]
    assert all(call[2] >= 20 for call in calls)
    assert all(call[3] == "session-a" for call in calls)
    assert {result.chunk.id for result in results} == {"dense", "lexical"}


def test_rrf_combines_rankings_and_deduplicates_chunks() -> None:
    first = IndexedChunk("first", "notes.txt", "first", None)
    shared = IndexedChunk("shared", "notes.txt", "shared", None)
    third = IndexedChunk("third", "notes.txt", "third", None)

    results = reciprocal_rank_fusion(
        [
            [
                SearchResult(chunk=first, score=0.9),
                SearchResult(chunk=shared, score=0.8),
            ],
            [
                SearchResult(chunk=shared, score=0.7),
                SearchResult(chunk=third, score=0.6),
            ],
        ],
        limit=10,
    )

    assert [result.chunk.id for result in results] == ["shared", "first", "third"]
    assert len(results) == 3


def test_reranker_changes_ordering_and_final_top_k_is_respected(monkeypatch) -> None:
    index = DocumentIndex(
        embedding_provider=FailingEmbeddingProvider(),
        reranker=PreferredTextReranker(),
    )
    first = IndexedChunk("first", "notes.txt", "first fused result", None)
    preferred = IndexedChunk("preferred", "notes.txt", "preferred reranker result", None)

    monkeypatch.setattr(
        index,
        "_semantic_search",
        lambda _query, _limit, _session_id: [
            SearchResult(chunk=first, score=0.9),
            SearchResult(chunk=preferred, score=0.8),
        ],
    )
    monkeypatch.setattr(
        index,
        "_lexical_search",
        lambda _query, _limit, _session_id: [],
    )

    results = index.search_with_scores("query", limit=1)

    assert [result.chunk.id for result in results] == ["preferred"]


def test_ollama_embedding_provider_posts_to_embed_endpoint(monkeypatch) -> None:
    requests = []

    def fake_urlopen(request, timeout: int) -> FakeEmbeddingResponse:
        requests.append((request, timeout))
        return FakeEmbeddingResponse(b'{"embeddings": [[1.0, 0.0], [0.0, 1.0]]}')

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.example.test")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "test-embed")
    monkeypatch.setattr(document_index_module, "urlopen", fake_urlopen)

    embeddings = OllamaEmbeddingProvider().embed(["first", "second"])

    request, timeout = requests[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "http://ollama.example.test/api/embed"
    assert timeout == 120
    assert payload == {"model": "test-embed", "input": ["first", "second"]}
    assert embeddings == [[1.0, 0.0], [0.0, 1.0]]


def build_test_pdf(text: str) -> bytes:
    stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode("utf-8")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode("utf-8")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]

    for index, value in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("utf-8"))
        pdf.extend(value)
        pdf.extend(b"\nendobj\n")

    startxref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("utf-8"))
    pdf.extend(b"0000000000 65535 f \n")

    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("utf-8"))

    pdf.extend(
        (
            f"trailer\n<< /Root 1 0 R /Size {len(objects) + 1} >>\n"
            f"startxref\n{startxref}\n%%EOF\n"
        ).encode("utf-8")
    )

    return bytes(pdf)
