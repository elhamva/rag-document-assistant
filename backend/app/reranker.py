from __future__ import annotations

import importlib.util
import logging

from backend.app import config
from backend.app.retrieval import SearchResult


logger = logging.getLogger(__name__)


class NoReranker:
    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        return candidates


MAX_SCORE_GAP = 8.0


class CrossEncoderReranker:
    def __init__(
        self, model_name: str = config.RERANKER_MODEL, max_score_gap: float = MAX_SCORE_GAP
    ) -> None:
        self.model_name = model_name
        self.max_score_gap = max_score_gap
        self._model = None

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        if len(candidates) < 2:
            return candidates
        return drop_far_below_best(self.score(query, candidates), self.max_score_gap)

    def score(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        """Rescore with the cross-encoder and sort, best first."""
        scores = self._load_model().predict([(query, result.chunk.text) for result in candidates])
        reranked = [
            SearchResult(chunk=result.chunk, score=float(score))
            for result, score in zip(candidates, scores)
        ]
        return sorted(reranked, key=lambda result: result.score, reverse=True)

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker model %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model


def drop_far_below_best(results: list[SearchResult], max_gap: float) -> list[SearchResult]:
    """Drop chunks whose score is far below the best one; expects results sorted best first."""
    if not results:
        return results
    best_score = results[0].score
    return [result for result in results if result.score >= best_score - max_gap]


def build_reranker():
    if not config.RERANKER_ENABLED:
        return NoReranker()

    if importlib.util.find_spec("sentence_transformers") is None:
        logger.warning("sentence-transformers is not installed, so reranking is off.")
        return NoReranker()

    return CrossEncoderReranker()
