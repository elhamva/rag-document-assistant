"""Compare retrieval strategies on the handbook, optionally buried in a ~500-page product catalogue.

Usage: python backend/scripts/evaluate_retrieval.py [--large]   (needs Ollama)
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

from eval_common import (
    WORDS_PER_PAGE,
    build_corpus,
    first_hit,
    hybrid,
    recall_and_mrr,
)

from backend.app.config import TOP_K
from backend.app.ollama import OllamaClient
from backend.app.reranker import CrossEncoderReranker, build_reranker
from backend.app.retrieval import DENSE_TOP_K, KEYWORD_TOP_K, dense_search, keyword_search, normalize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large", action="store_true", help="add the synthetic ~500-page catalogue")
    args = parser.parse_args()

    ollama = OllamaClient()
    corpus = build_corpus(ollama, args.large)
    chunks = corpus.chunks

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

    ranks = {name: defaultdict(list) for name in strategies}
    misses = []
    seconds = defaultdict(float)
    for question in corpus.questions:
        started = time.perf_counter()
        query_vector = normalize(ollama.embed_query(question.text))
        seconds["query embedding"] += time.perf_counter() - started
        for name, search in strategies.items():
            started = time.perf_counter()
            results = search(question.text, query_vector)
            seconds[name] += time.perf_counter() - started
            rank = first_hit(results, question.expected)
            ranks[name][question.group].append(rank)
        if rank is None or rank > TOP_K:  # the last strategy is the full pipeline
            misses.append(question.text)

    groups = list(ranks["dense"])
    print(
        f"{len(chunks)} chunks, {corpus.words:,} words (~{corpus.words // WORDS_PER_PAGE} pages), "
        f"embedded in {corpus.embed_seconds:.1f} s; {len(corpus.questions)} questions, k = {TOP_K}\n"
    )
    header = "".join(f"{group + ' R@' + str(TOP_K):>18}{'MRR':>7}" for group in groups)
    print(f"{'strategy':<18}{header}{'ms/query':>10}")
    for name, by_group in ranks.items():
        cells = "".join(f"{recall:>18.2f}{mrr:>7.2f}" for recall, mrr in (recall_and_mrr(by_group[g], TOP_K) for g in groups))
        print(f"{name:<18}{cells}{1000 * seconds[name] / len(corpus.questions):>10.0f}")
    print(f"\nQuery embedding: {1000 * seconds['query embedding'] / len(corpus.questions):.0f} ms/query")
    for question in misses:
        print(f"Missed by {name}: {question}")


if __name__ == "__main__":
    main()
