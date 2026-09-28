from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from backend.app.ingestion import Chunk


DENSE_TOP_K = 20
KEYWORD_TOP_K = 20
RERANK_CANDIDATES = 20
MIN_DENSE_SIMILARITY = 0.5
RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75
STOP_WORDS = set(
    "a an and are as at be by can do does for from has have how i in is it its of on or "
    "that the their this to was what when where which who why will with you your".split()
)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]: ...


class DocumentIndex:
    def __init__(self, embedder: Embedder, reranker: Reranker) -> None:
        self.embedder = embedder
        self.reranker = reranker
        self._sessions: dict[str, dict[str, list[Chunk]]] = {}

    def documents(self, session_id: str) -> dict[str, list[Chunk]]:
        return dict(self._sessions.get(session_id, {}))

    def chunks(self, session_id: str) -> list[Chunk]:
        return [chunk for chunks in self.documents(session_id).values() for chunk in chunks]

    def replace(self, session_id: str, documents: dict[str, list[Chunk]]) -> None:
        new_chunks = [
            chunk for chunks in documents.values() for chunk in chunks if chunk.embedding is None
        ]
        if new_chunks:
            vectors = self.embedder.embed_documents([chunk.text for chunk in new_chunks])
            for chunk, vector in zip(new_chunks, vectors):
                chunk.embedding = normalize(vector)

        self._sessions[session_id] = documents

    def search(self, session_id: str, query: str, top_k: int) -> list[SearchResult]:
        chunks = self.chunks(session_id)
        if not chunks:
            return []

        query_vector = normalize(self.embedder.embed_query(query))
        candidates = reciprocal_rank_fusion(
            [
                dense_search(chunks, query_vector, DENSE_TOP_K),
                keyword_search(chunks, query, KEYWORD_TOP_K),
            ]
        )[:RERANK_CANDIDATES]
        return self.reranker.rerank(query, candidates)[:top_k]


def dense_search(chunks: list[Chunk], query_vector: list[float], limit: int) -> list[SearchResult]:
    results = []
    for chunk in chunks:
        score = sum(a * b for a, b in zip(query_vector, chunk.embedding))
        if score >= MIN_DENSE_SIMILARITY:
            results.append(SearchResult(chunk, score))
    return sorted(results, key=lambda result: result.score, reverse=True)[:limit]


def keyword_search(chunks: list[Chunk], query: str, limit: int) -> list[SearchResult]:
    query_terms = set(tokenize(query))
    documents = [tokenize(chunk.text) for chunk in chunks]
    average_length = sum(len(document) for document in documents) / len(documents) or 1
    document_frequency = Counter(term for document in documents for term in set(document))

    results = []
    for chunk, document in zip(chunks, documents):
        term_counts = Counter(document)
        score = 0.0
        for term in query_terms:
            count = term_counts[term]
            if count == 0:
                continue
            frequency = document_frequency[term]
            idf = math.log(1 + (len(documents) - frequency + 0.5) / (frequency + 0.5))
            length_norm = 1 - BM25_B + BM25_B * len(document) / average_length
            score += idf * count * (BM25_K1 + 1) / (count + BM25_K1 * length_norm)
        if score > 0:
            results.append(SearchResult(chunk, score))

    return sorted(results, key=lambda result: result.score, reverse=True)[:limit]


def reciprocal_rank_fusion(result_lists: list[list[SearchResult]], k: int = RRF_K) -> list[SearchResult]:
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            scores[result.chunk.id] = scores.get(result.chunk.id, 0.0) + 1 / (k + rank)
            chunks[result.chunk.id] = result.chunk

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return [SearchResult(chunks[chunk_id], scores[chunk_id]) for chunk_id in ranked_ids]


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"\w+", text.lower()) if token not in STOP_WORDS]


def normalize(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in vector))
    return [value / length for value in vector] if length else vector
