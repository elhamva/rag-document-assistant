from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CHUNK_WORD_COUNT = 180
CHUNK_OVERLAP_WORD_COUNT = 40
MIN_RELEVANCE_SCORE = 0.2
MIN_EMBEDDING_SIMILARITY = 0.45
DEFAULT_DENSE_TOP_K = 20
DEFAULT_LEXICAL_TOP_K = 20
DEFAULT_CANDIDATE_TOP_K = 20
RRF_RANK_CONSTANT = 60
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
OLLAMA_BASE_URL_ENV_VAR = "OLLAMA_BASE_URL"
OLLAMA_EMBED_MODEL_ENV_VAR = "OLLAMA_EMBED_MODEL"
DENSE_TOP_K_ENV_VAR = "RAG_DENSE_TOP_K"
LEXICAL_TOP_K_ENV_VAR = "RAG_LEXICAL_TOP_K"
CANDIDATE_TOP_K_ENV_VAR = "RAG_CANDIDATE_TOP_K"
RERANKER_MODEL_ENV_VAR = "RAG_RERANKER_MODEL"
RERANKER_ENABLED_ENV_VAR = "RAG_RERANKER_ENABLED"
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "her",
    "his",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


class DocumentProcessingError(ValueError):
    pass


@dataclass(frozen=True)
class TextPage:
    text: str
    page_number: int | None = None


@dataclass(frozen=True)
class IndexedChunk:
    id: str
    filename: str
    text: str
    page_number: int | None
    embedding: list[float] | None = None


@dataclass(frozen=True)
class SearchResult:
    chunk: IndexedChunk
    score: float


class DocumentIndex:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._chunks_by_session: dict[str, list[IndexedChunk]] = {}
        self._embedding_available_by_session: dict[str, bool] = {}
        self._embedding_provider = embedding_provider or OllamaEmbeddingProvider()
        self._reranker = reranker or CrossEncoderReranker()
        self.embedding_available = False

    def replace_chunks(
        self,
        chunks: list[IndexedChunk],
        session_id: str = "default",
    ) -> None:
        self._chunks_by_session[session_id] = self._embed_chunks(chunks, session_id)

    def list_chunks(self, session_id: str = "default") -> list[IndexedChunk]:
        return list(self._chunks_by_session.get(session_id, []))

    def clear(self, session_id: str | None = None) -> None:
        if session_id is not None:
            self._chunks_by_session.pop(session_id, None)
            self._embedding_available_by_session.pop(session_id, None)
            self.embedding_available = False
            return

        self._chunks_by_session = {}
        self._embedding_available_by_session = {}
        self.embedding_available = False

    def search(
        self,
        query: str,
        limit: int = 5,
        session_id: str = "default",
    ) -> list[IndexedChunk]:
        return [
            result.chunk
            for result in self.search_with_scores(query, limit, session_id)
        ]

    def search_with_scores(
        self,
        query: str,
        limit: int = 5,
        session_id: str = "default",
    ) -> list[SearchResult]:
        final_limit = max(1, limit)
        dense_limit = max(
            final_limit,
            get_positive_int_env(DENSE_TOP_K_ENV_VAR, DEFAULT_DENSE_TOP_K),
        )
        lexical_limit = max(
            final_limit,
            get_positive_int_env(LEXICAL_TOP_K_ENV_VAR, DEFAULT_LEXICAL_TOP_K),
        )
        candidate_limit = max(
            final_limit,
            get_positive_int_env(CANDIDATE_TOP_K_ENV_VAR, DEFAULT_CANDIDATE_TOP_K),
        )
        semantic_results = self._semantic_search(query, dense_limit, session_id)
        lexical_results = self._lexical_search(query, lexical_limit, session_id)
        fused_results = reciprocal_rank_fusion(
            [semantic_results, lexical_results],
            limit=candidate_limit,
        )
        reranked_results = self._reranker.rerank(query, fused_results)
        return reranked_results[:final_limit]

    def _embed_chunks(
        self,
        chunks: list[IndexedChunk],
        session_id: str,
    ) -> list[IndexedChunk]:
        if not chunks:
            self._set_embedding_available(session_id, False)
            return []

        try:
            embeddings = self._embedding_provider.embed([chunk.text for chunk in chunks])
        except EmbeddingError:
            self._set_embedding_available(session_id, False)
            return list(chunks)

        if len(embeddings) != len(chunks):
            self._set_embedding_available(session_id, False)
            return list(chunks)

        self._set_embedding_available(session_id, True)
        return [
            IndexedChunk(
                id=chunk.id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                text=chunk.text,
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

    def _set_embedding_available(self, session_id: str, value: bool) -> None:
        self._embedding_available_by_session[session_id] = value
        self.embedding_available = value

    def _semantic_search(
        self,
        query: str,
        limit: int,
        session_id: str,
    ) -> list[SearchResult]:
        if not self._embedding_available_by_session.get(session_id, False):
            return []

        try:
            query_embeddings = self._embedding_provider.embed([query])
        except EmbeddingError:
            return []

        if not query_embeddings:
            return []

        query_embedding = query_embeddings[0]
        results = []

        for chunk in self._chunks_by_session.get(session_id, []):
            if chunk.embedding is None:
                continue

            score = cosine_similarity(query_embedding, chunk.embedding)
            if score >= MIN_EMBEDDING_SIMILARITY:
                results.append(SearchResult(chunk=chunk, score=round(score, 4)))

        results.sort(key=lambda result: result.score, reverse=True)
        return results[:limit]

    def _lexical_search(
        self,
        query: str,
        limit: int,
        session_id: str,
    ) -> list[SearchResult]:
        query_terms = set(tokenize_for_search(query))
        if not query_terms:
            return []

        scored_chunks = []
        for chunk in self._chunks_by_session.get(session_id, []):
            score = len(query_terms.intersection(tokenize_for_search(chunk.text)))
            relevance_score = round(score / len(query_terms), 4)
            if relevance_score >= MIN_RELEVANCE_SCORE:
                scored_chunks.append(
                    SearchResult(
                        chunk=chunk,
                        score=relevance_score,
                    )
                )

        scored_chunks.sort(key=lambda result: result.score, reverse=True)
        return scored_chunks[:limit]


class EmbeddingError(RuntimeError):
    pass


class RerankerError(RuntimeError):
    pass


class EmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class Reranker:
    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    def __init__(
        self,
        model_name: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv(
            RERANKER_MODEL_ENV_VAR,
            DEFAULT_RERANKER_MODEL,
        )
        self._enabled_override = enabled
        self._model = None

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        if len(candidates) <= 1 or not self._is_enabled():
            return candidates

        model = self._load_model()
        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        scores = model.predict(pairs)
        reranked = [
            SearchResult(chunk=candidate.chunk, score=round(float(score), 4))
            for candidate, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda result: result.score, reverse=True)
        return reranked

    def _is_enabled(self) -> bool:
        if self._enabled_override is not None:
            return self._enabled_override

        value = os.getenv(RERANKER_ENABLED_ENV_VAR, "auto").strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True

        if value in {"0", "false", "no", "off"}:
            return False

        return is_sentence_transformers_available()

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerError(
                "Cross-encoder reranking requires sentence-transformers. "
                "Install backend requirements or set RAG_RERANKER_ENABLED=0."
            ) from exc

        self._model = CrossEncoder(self.model_name)
        return self._model


class OllamaEmbeddingProvider(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        base_url = os.getenv(OLLAMA_BASE_URL_ENV_VAR, DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        model = os.getenv(OLLAMA_EMBED_MODEL_ENV_VAR, DEFAULT_OLLAMA_EMBED_MODEL)
        request = Request(
            f"{base_url}/api/embed",
            data=json.dumps(
                {
                    "model": model,
                    "input": texts,
                }
            ).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise EmbeddingError(format_embedding_http_error(exc)) from exc
        except (OSError, URLError, json.JSONDecodeError) as exc:
            raise EmbeddingError("Ollama embeddings are not available.") from exc

        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingError("Ollama returned an invalid embedding response.")

        return [normalize_embedding(embedding) for embedding in embeddings]


document_index = DocumentIndex()


def process_document(filename: str, content: bytes) -> list[IndexedChunk]:
    extension = Path(filename).suffix.lower()

    if extension == ".txt":
        pages = [TextPage(text=decode_text(content))]
    elif extension == ".pdf":
        pages = extract_pdf_pages(content)
    else:
        raise DocumentProcessingError("Unsupported file type.")

    chunks: list[IndexedChunk] = []
    for page in pages:
        for chunk_text in split_text(page.text):
            chunks.append(
                IndexedChunk(
                    id=build_chunk_id(filename, page.page_number, len(chunks), chunk_text),
                    filename=filename,
                    page_number=page.page_number,
                    text=chunk_text,
                )
            )

    if not chunks:
        raise DocumentProcessingError("No readable text found.")

    return chunks


def decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise DocumentProcessingError("Text file could not be decoded.")


def extract_pdf_pages(content: bytes) -> list[TextPage]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentProcessingError(
            "PDF support requires pypdf. Install backend requirements."
        ) from exc

    try:
        reader = PdfReader(BytesIO(content))
    except Exception as exc:
        raise DocumentProcessingError("PDF text could not be extracted.") from exc

    pages: list[TextPage] = []

    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            raise DocumentProcessingError("PDF text could not be extracted.") from exc

        if text.strip():
            pages.append(TextPage(text=text, page_number=index))

    return pages


def split_text(text: str) -> list[str]:
    words = normalize_whitespace(text).split()
    if not words:
        return []

    chunks: list[str] = []
    step = CHUNK_WORD_COUNT - CHUNK_OVERLAP_WORD_COUNT

    for start in range(0, len(words), step):
        chunk_words = words[start : start + CHUNK_WORD_COUNT]
        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if start + CHUNK_WORD_COUNT >= len(words):
            break

    return chunks


def build_chunk_id(
    filename: str,
    page_number: int | None,
    chunk_index: int,
    text: str,
) -> str:
    value = f"{filename}:{page_number}:{chunk_index}:{text}"
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"chunk-{digest}"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def tokenize_for_search(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOP_WORDS]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def normalize_embedding(value: object) -> list[float]:
    if not isinstance(value, list):
        raise EmbeddingError("Ollama returned an invalid embedding response.")

    embedding = []
    for item in value:
        if not isinstance(item, (int, float)):
            raise EmbeddingError("Ollama returned an invalid embedding response.")

        embedding.append(float(item))

    if not embedding:
        raise EmbeddingError("Ollama returned an invalid embedding response.")

    return embedding


def reciprocal_rank_fusion(
    result_sets: list[list[SearchResult]],
    limit: int,
    rank_constant: int = RRF_RANK_CONSTANT,
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, IndexedChunk] = {}
    first_seen_by_id: dict[str, int] = {}
    next_seen_index = 0

    for results in result_sets:
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.id
            if chunk_id not in chunks_by_id:
                chunks_by_id[chunk_id] = result.chunk
                first_seen_by_id[chunk_id] = next_seen_index
                next_seen_index += 1

            scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                1.0 / (rank_constant + rank)
            )

    ranked_chunk_ids = sorted(
        scores,
        key=lambda chunk_id: (-scores[chunk_id], first_seen_by_id[chunk_id]),
    )
    return [
        SearchResult(
            chunk=chunks_by_id[chunk_id],
            score=round(scores[chunk_id], 4),
        )
        for chunk_id in ranked_chunk_ids[: max(1, limit)]
    ]


def get_positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default

    try:
        return max(1, int(value))
    except ValueError:
        return default


def is_sentence_transformers_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def format_embedding_http_error(exc: HTTPError) -> str:
    try:
        payload = json.loads(exc.read().decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        payload = None

    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return f"Ollama embeddings returned HTTP {exc.code}: {payload['error']}"

    return f"Ollama embeddings returned HTTP {exc.code}."
