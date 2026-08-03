"""Lazy local BGE embedding adapter."""

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Port implemented by local embedding models and deterministic test doubles."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Create one normalized vector per input text."""


class BGEEmbeddingProvider:
    """Loads `BAAI/bge-small-en-v1.5` only when the first indexing job needs it."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed texts locally with normalized vectors suitable for cosine similarity."""
        if not texts:
            return []
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()
