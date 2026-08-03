"""In-memory BM25 indexes rebuilt from durable chunk metadata on service startup."""

import re

from rank_bm25 import BM25Okapi

from backend.app.domain.entities import ChunkRecord


class BM25IndexManager:
    """Builds per-document lexical indexes without exposing retrieval operations yet."""

    def __init__(self) -> None:
        self._indexes: dict[str, BM25Okapi] = {}

    def rebuild_document_index(self, document_id: str, chunks: list[ChunkRecord]) -> None:
        """Replace one document's lexical index after successful chunk generation."""
        if chunks:
            self._indexes[document_id] = BM25Okapi([self._tokenize(chunk.text) for chunk in chunks])
        else:
            self._indexes.pop(document_id, None)

    def has_index(self, document_id: str) -> bool:
        """Report whether an in-process lexical index exists for a document."""
        return document_id in self._indexes

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())
