import pytest

from backend.app.ingestion import Chunk
from backend.app.ollama import OllamaError
from backend.app.retrieval import (
    DocumentIndex,
    SearchResult,
    keyword_search,
    reciprocal_rank_fusion,
)


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(id=chunk_id, filename="notes.txt", page=None, text=text)


class PreferReranker:
    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        return sorted(candidates, key=lambda result: "preferred" not in result.chunk.text)


def test_dense_search_finds_related_chunk_without_shared_keyword(index: DocumentIndex) -> None:
    index.replace("s", {"doc": [chunk("fruit", "Bananas are yellow."), chunk("car", "The car needs service.")]})

    results = index.search("s", "automobile upkeep", top_k=1)

    assert [result.chunk.id for result in results] == ["car"]


def test_keyword_search_ranks_rare_terms_higher() -> None:
    chunks = [
        chunk("common", "The lamp uses power. The lamp is bright."),
        chunk("rare", "The lamp model XR-200 uses 40 watts."),
        chunk("other", "The lamp is white."),
    ]

    results = keyword_search(chunks, "lamp XR-200", limit=3)

    assert results[0].chunk.id == "rare"


def test_keyword_search_ignores_stop_words() -> None:
    assert keyword_search([chunk("a", "What is the answer")], "what is the", limit=5) == []


def test_rrf_merges_rankings_and_removes_duplicates() -> None:
    first, shared, third = chunk("first", "1"), chunk("shared", "2"), chunk("third", "3")

    results = reciprocal_rank_fusion(
        [
            [SearchResult(first, 0.9), SearchResult(shared, 0.8)],
            [SearchResult(shared, 7.0), SearchResult(third, 6.0)],
        ]
    )

    assert [result.chunk.id for result in results] == ["shared", "first", "third"]


def test_reranker_decides_final_order_and_top_k_is_respected(embedder) -> None:
    index = DocumentIndex(embedder=embedder, reranker=PreferReranker())
    index.replace("s", {"doc": [chunk("plain", "car manual"), chunk("preferred", "car preferred manual")]})

    results = index.search("s", "car manual", top_k=1)

    assert [result.chunk.id for result in results] == ["preferred"]


def test_replace_only_embeds_new_chunks(index: DocumentIndex, embedder) -> None:
    old = [chunk("old", "car text")]
    index.replace("s", {"old": old})
    index.replace("s", {"old": old, "new": [chunk("new", "banana text")]})

    assert embedder.embedded_texts == ["car text", "banana text"]


def test_failed_embedding_keeps_previous_index(index: DocumentIndex, embedder) -> None:
    index.replace("s", {"old": [chunk("old", "car text")]})
    embedder.available = False

    with pytest.raises(OllamaError):
        index.replace("s", {"new": [chunk("new", "banana text")]})

    assert [c.id for c in index.chunks("s")] == ["old"]


def test_sessions_are_isolated(index: DocumentIndex) -> None:
    index.replace("alpha", {"doc": [chunk("a", "car text")]})

    assert index.search("beta", "car", top_k=4) == []
    assert index.search("alpha", "car", top_k=4)[0].chunk.id == "a"
