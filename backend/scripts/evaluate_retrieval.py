from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

config = importlib.import_module("backend.app.config")
ingestion = importlib.import_module("backend.app.ingestion")
ollama_module = importlib.import_module("backend.app.ollama")
reranker_module = importlib.import_module("backend.app.reranker")
retrieval = importlib.import_module("backend.app.retrieval")

TOP_K = config.TOP_K
Chunk = ingestion.Chunk
parse_document = ingestion.parse_document
OllamaClient = ollama_module.OllamaClient
CrossEncoderReranker = reranker_module.CrossEncoderReranker
build_reranker = reranker_module.build_reranker
DENSE_TOP_K = retrieval.DENSE_TOP_K
KEYWORD_TOP_K = retrieval.KEYWORD_TOP_K
RERANK_CANDIDATES = retrieval.RERANK_CANDIDATES
SearchResult = retrieval.SearchResult
dense_search = retrieval.dense_search
keyword_search = retrieval.keyword_search
normalize = retrieval.normalize
reciprocal_rank_fusion = retrieval.reciprocal_rank_fusion

DOCUMENT = ROOT / "sample_docs" / "brightline_service_handbook.md"
QUESTIONS = Path(__file__).with_name("eval_questions.json")


def main() -> None:
    ollama = OllamaClient()
    chunks = parse_document(DOCUMENT.name, DOCUMENT.read_bytes())
    for chunk, vector in zip(chunks, ollama.embed_documents([chunk.text for chunk in chunks])):
        chunk.embedding = normalize(vector)
    questions = json.loads(QUESTIONS.read_text())

    strategies = {
        "dense": lambda query, vector: dense_search(chunks, vector, DENSE_TOP_K),
        "bm25": lambda query, vector: keyword_search(chunks, query, KEYWORD_TOP_K),
        "hybrid (RRF)": lambda query, vector: hybrid(chunks, query, vector),
    }
    reranker = build_reranker()
    if isinstance(reranker, CrossEncoderReranker):
        strategies["hybrid + rerank"] = lambda query, vector: reranker.rerank(
            query, hybrid(chunks, query, vector)
        )
    else:
        print("Reranker unavailable (install sentence-transformers), skipping it.\n")

    ranks: dict[str, list[Optional[int]]] = {name: [] for name in strategies}
    for item in questions:
        query_vector = normalize(ollama.embed_query(item["question"]))
        for name, search in strategies.items():
            results = search(item["question"], query_vector)
            ranks[name].append(first_hit(results, item["expected"]))

    print(f"{len(questions)} questions, {len(chunks)} chunks, k = {TOP_K}\n")
    print(f"{'strategy':<18}{'Recall@' + str(TOP_K):>10}{'MRR':>8}")
    for name, found in ranks.items():
        hits = [rank for rank in found if rank is not None and rank <= TOP_K]
        recall = len(hits) / len(found)
        mrr = sum(1 / rank for rank in hits) / len(found)
        print(f"{name:<18}{recall:>10.2f}{mrr:>8.2f}")


def hybrid(chunks: list[Chunk], query: str, query_vector: list[float]) -> list[SearchResult]:
    return reciprocal_rank_fusion(
        [
            dense_search(chunks, query_vector, DENSE_TOP_K),
            keyword_search(chunks, query, KEYWORD_TOP_K),
        ]
    )[:RERANK_CANDIDATES]


def first_hit(results: list[SearchResult], expected: str) -> Optional[int]:
    for rank, result in enumerate(results, start=1):
        if expected.lower() in result.chunk.text.lower():
            return rank
    return None


if __name__ == "__main__":
    main()
