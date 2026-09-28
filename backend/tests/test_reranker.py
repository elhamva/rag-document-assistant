from backend.app.ingestion import Chunk
from backend.app.reranker import CrossEncoderReranker
from backend.app.retrieval import SearchResult


class FakeCrossEncoder:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self.scores[text] for _query, text in pairs]


def candidates(*texts: str) -> list[SearchResult]:
    return [SearchResult(Chunk(id=text, filename="a.txt", page=None, text=text), 0.0) for text in texts]


def reranker_with(scores: dict[str, float]) -> CrossEncoderReranker:
    reranker = CrossEncoderReranker()
    reranker._model = FakeCrossEncoder(scores)
    return reranker


def test_orders_by_cross_encoder_score_and_drops_clear_noise() -> None:
    reranker = reranker_with({"weak": 1.0, "best": 3.5, "noise": -11.0})

    results = reranker.rerank("question", candidates("weak", "best", "noise"))

    assert [result.chunk.id for result in results] == ["best", "weak"]


def test_keeps_everything_when_no_chunk_clearly_matches() -> None:
    reranker = reranker_with({"a": -11.0, "b": -10.5})

    results = reranker.rerank("summarize the document", candidates("a", "b"))

    assert [result.chunk.id for result in results] == ["b", "a"]
