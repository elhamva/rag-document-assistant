"""Sweep chunk size and overlap, the dense similarity cut-off and the reranker score gap.

Runs on the handbook plus the ~500-page catalogue, so the numbers are not saturated.
Usage: python backend/scripts/tune_parameters.py [--sweep chunking|similarity|gap]
(needs Ollama and sentence-transformers)
"""

from __future__ import annotations

import argparse
import statistics

from eval_common import Corpus, build_corpus, first_hit, hybrid, recall_and_mrr, unanswerable_questions

from backend.app.config import TOP_K
from backend.app.ollama import OllamaClient
from backend.app.reranker import CrossEncoderReranker, drop_far_below_best
from backend.app.retrieval import DENSE_TOP_K, dense_search, normalize

CHUNKINGS = [(100, 20), (180, 0), (180, 40), (300, 60), (500, 100)]
SIMILARITY_CUTOFFS = [0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
SCORE_GAPS = [2.0, 4.0, 6.0, 8.0, 10.0, float("inf")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", choices=["chunking", "similarity", "gap"], help="run only one sweep")
    args = parser.parse_args()

    ollama = OllamaClient()
    reranker = CrossEncoderReranker()
    if args.sweep in (None, "chunking"):
        sweep_chunking(ollama, reranker)
    if args.sweep != "chunking":
        corpus = build_corpus(ollama, large=True)
        if args.sweep in (None, "similarity"):
            sweep_similarity_cutoff(ollama, reranker, corpus)
        if args.sweep in (None, "gap"):
            sweep_score_gap(ollama, reranker, corpus)


def sweep_chunking(ollama: OllamaClient, reranker: CrossEncoderReranker) -> None:
    print("Chunk size and overlap (hybrid + rerank)")
    print(f"{'words/overlap':<15}{'chunks':>8}{'R@' + str(TOP_K):>7}{'MRR':>7}{'context words':>15}")
    for chunk_words, overlap_words in CHUNKINGS:
        corpus = build_corpus(ollama, large=True, chunk_words=chunk_words, overlap_words=overlap_words)
        ranks, context_words = [], []
        for question in corpus.questions:
            vector = normalize(ollama.embed_query(question.text))
            results = reranker.rerank(question.text, hybrid(corpus.chunks, question.text, vector))[:TOP_K]
            ranks.append(first_hit(results, question.expected))
            context_words.append(sum(len(result.chunk.text.split()) for result in results))
        recall, mrr = recall_and_mrr(ranks, TOP_K)
        print(
            f"{f'{chunk_words}/{overlap_words}':<15}{len(corpus.chunks):>8}{recall:>7.2f}{mrr:>7.2f}"
            f"{statistics.mean(context_words):>15.0f}"
        )


def sweep_similarity_cutoff(ollama: OllamaClient, reranker: CrossEncoderReranker, corpus: Corpus) -> None:
    answerable = [(q, normalize(ollama.embed_query(q.text))) for q in corpus.questions]
    unanswerable = [(q, normalize(ollama.embed_query(q))) for q in unanswerable_questions()]

    gold = [
        max(similarity(vector, c.embedding) for c in corpus.chunks if q.expected.lower() in c.text.lower())
        for q, vector in answerable
    ]
    best_unanswerable = [max(similarity(vector, c.embedding) for c in corpus.chunks) for _q, vector in unanswerable]
    print("\nCosine similarity of the correct chunk (answerable) vs. the closest chunk (unanswerable)")
    print(f"  correct chunk:       min {min(gold):.2f}  10th pct {percentile(gold, 10):.2f}  median {statistics.median(gold):.2f}")
    print(f"  unanswerable, best:  min {min(best_unanswerable):.2f}  median {statistics.median(best_unanswerable):.2f}  max {max(best_unanswerable):.2f}")

    print("\nDense similarity cut-off")
    print(f"{'cut-off':<10}{'dense R@20':>11}{'final R@' + str(TOP_K):>11}{'MRR':>7}{'dense hits, unanswerable':>26}")
    for cutoff in SIMILARITY_CUTOFFS:
        dense_ranks, final_ranks = [], []
        for question, vector in answerable:
            dense_ranks.append(first_hit(dense_search(corpus.chunks, vector, DENSE_TOP_K, cutoff), question.expected))
            candidates = hybrid(corpus.chunks, question.text, vector, cutoff)
            final_ranks.append(first_hit(reranker.rerank(question.text, candidates)[:TOP_K], question.expected))
        noise = statistics.mean(
            len(dense_search(corpus.chunks, vector, DENSE_TOP_K, cutoff)) for _q, vector in unanswerable
        )
        dense_recall, _ = recall_and_mrr(dense_ranks, DENSE_TOP_K)
        recall, mrr = recall_and_mrr(final_ranks, TOP_K)
        print(f"{cutoff:<10}{dense_recall:>11.2f}{recall:>11.2f}{mrr:>7.2f}{noise:>26.1f}")


def sweep_score_gap(ollama: OllamaClient, reranker: CrossEncoderReranker, corpus: Corpus) -> None:
    def scored(question: str) -> list:
        return reranker.score(question, hybrid(corpus.chunks, question, normalize(ollama.embed_query(question))))

    answerable = [(q, scored(q.text)) for q in corpus.questions]
    unanswerable = [scored(q) for q in unanswerable_questions()]

    gaps = []
    for question, results in answerable:
        rank = first_hit(results, question.expected)
        if rank is not None:
            gaps.append(results[0].score - results[rank - 1].score)
    print("\nReranker score of the correct chunk, below the best chunk")
    print(f"  median {statistics.median(gaps):.1f}  90th pct {percentile(gaps, 90):.1f}  max {max(gaps):.1f}")

    print("\nReranker score gap (chunks kept, of at most " + str(TOP_K) + ")")
    print(f"{'max gap':<10}{'R@' + str(TOP_K):>7}{'MRR':>7}{'kept, answerable':>18}{'kept, unanswerable':>20}")
    for gap in SCORE_GAPS:
        kept = [drop_far_below_best(results, gap)[:TOP_K] for _q, results in answerable]
        ranks = [first_hit(results, q.expected) for (q, _r), results in zip(answerable, kept)]
        noise = [len(drop_far_below_best(results, gap)[:TOP_K]) for results in unanswerable]
        recall, mrr = recall_and_mrr(ranks, TOP_K)
        print(
            f"{gap:<10}{recall:>7.2f}{mrr:>7.2f}{statistics.mean(len(k) for k in kept):>18.1f}"
            f"{statistics.mean(noise):>20.1f}"
        )


def similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def percentile(values: list[float], pct: int) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


if __name__ == "__main__":
    main()
