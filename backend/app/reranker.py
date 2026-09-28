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
    def __init__(self, model_name: str = config.RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        if len(candidates) < 2:
            return candidates

        scores = self._load_model().predict([(query, result.chunk.text) for result in candidates])
        reranked = [
            SearchResult(chunk=result.chunk, score=float(score))
            for result, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda result: result.score, reverse=True)
        best_score = reranked[0].score
        return [result for result in reranked if result.score >= best_score - MAX_SCORE_GAP]

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker model %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model


def build_reranker():
    if not config.RERANKER_ENABLED:
        return NoReranker()

    if importlib.util.find_spec("sentence_transformers") is None:
        logger.warning("sentence-transformers is not installed, so reranking is off.")
        return NoReranker()

    return CrossEncoderReranker()
