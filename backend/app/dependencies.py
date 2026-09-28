from functools import lru_cache

from backend.app.ollama import OllamaClient
from backend.app.reranker import build_reranker
from backend.app.retrieval import DocumentIndex


@lru_cache
def get_llm() -> OllamaClient:
    return OllamaClient()


@lru_cache
def get_index() -> DocumentIndex:
    return DocumentIndex(embedder=get_llm(), reranker=build_reranker())
