from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.document_index import (
    DocumentIndex,
    EmbeddingProvider,
    IndexedChunk,
    Reranker,
    SearchResult,
)


class EvalEmbeddingProvider(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        text = text.lower()
        return [
            float(any(term in text for term in ("agreement", "contract", "renewal"))),
            float(any(term in text for term in ("salary", "compensation", "bands"))),
            float(any(term in text for term in ("vacation", "pto", "paid time off"))),
        ]


class PassthroughReranker(Reranker):
    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        return candidates


def main() -> None:
    index = DocumentIndex(
        embedding_provider=EvalEmbeddingProvider(),
        reranker=PassthroughReranker(),
    )
    chunks = [
        IndexedChunk(
            id="contract",
            filename="hr.txt",
            page_number=None,
            text="The contract renewal date is May 12.",
        ),
        IndexedChunk(
            id="salary",
            filename="hr.txt",
            page_number=None,
            text="Salary details include compensation bands.",
        ),
        IndexedChunk(
            id="pto",
            filename="hr.txt",
            page_number=None,
            text="Vacation policy covers paid time off requests.",
        ),
    ]
    examples = [
        ("agreement renewal", "contract"),
        ("compensation bands", "salary"),
        ("paid time off", "pto"),
    ]

    index.replace_chunks(chunks, session_id="eval")
    hits = 0
    k = 3

    for query, expected_chunk_id in examples:
        result_ids = [
            chunk.id for chunk in index.search(query, limit=k, session_id="eval")
        ]
        matched = expected_chunk_id in result_ids
        hits += int(matched)
        print(
            f"query={query!r} expected={expected_chunk_id} "
            f"results={result_ids} hit={matched}"
        )

    recall = hits / len(examples)
    print(f"Recall@{k}: {recall:.2f} ({hits}/{len(examples)})")


if __name__ == "__main__":
    main()
