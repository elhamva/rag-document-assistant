from __future__ import annotations

from typing import Optional

import httpx

from backend.app import config


EMBED_BATCH_SIZE = 64


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: str = config.OLLAMA_BASE_URL,
        model: str = config.OLLAMA_MODEL,
        embed_model: str = config.OLLAMA_EMBED_MODEL,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.embed_model = embed_model
        self._use_prefixes = embed_model.startswith("nomic-embed")

    def generate(
        self,
        prompt: str,
        timeout: float = config.OLLAMA_TIMEOUT_SECONDS,
        json_schema: Optional[dict] = None,
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
        if json_schema is not None:
            payload["format"] = json_schema  # Ollama constrains decoding to this schema
        return self._post("/api/generate", payload, timeout)["response"]

    def with_model(self, model: str) -> "OllamaClient":
        return OllamaClient(base_url=self.base_url, model=model, embed_model=self.embed_model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefix = "search_document: " if self._use_prefixes else ""
        vectors = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = [prefix + text for text in texts[start : start + EMBED_BATCH_SIZE]]
            vectors.extend(self._embed(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        prefix = "search_query: " if self._use_prefixes else ""
        return self._embed([prefix + text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._post("/api/embed", {"model": self.embed_model, "input": texts})["embeddings"]

    def _post(
        self, path: str, payload: dict, timeout: float = config.OLLAMA_TIMEOUT_SECONDS
    ) -> dict:
        try:
            response = httpx.post(
                self.base_url + path,
                json=payload,
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.base_url}. Is `ollama serve` running?"
            ) from exc

        if response.status_code != 200:
            try:
                detail = response.json()["error"]
            except (ValueError, KeyError, TypeError):
                detail = response.text
            raise OllamaError(f"Ollama returned HTTP {response.status_code}: {detail}")

        return response.json()
