"""Lazy, cached adapter for the local MS MARCO cross-encoder."""

from threading import Lock
from typing import Protocol


class CrossEncoderProvider(Protocol):
    """Port allowing the reranking service to be tested without model downloads."""

    def predict(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        """Return one relevance logit for every query/passage pair."""


class LocalCrossEncoder:
    """Loads `cross-encoder/ms-marco-MiniLM-L-6-v2` once and reuses it for all requests."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._load_lock = Lock()

    def predict(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        """Run batched local cross-encoder inference, loading the model exactly once."""
        if not pairs:
            return []
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(self.model_name)
        scores = self._model.predict(
            pairs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        return [float(score) for score in scores]
