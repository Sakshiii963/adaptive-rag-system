"""Durable-source, in-memory BM25 keyword index for local hybrid retrieval."""

import re

from rank_bm25 import BM25Okapi

from backend.app.domain.entities import ChunkRecord, RetrievalCandidate, RetrievalFilters


class BM25IndexManager:
    """Maintains a global lexical index rebuilt from SQLite-backed document chunks."""

    def __init__(self) -> None:
        self._chunks: dict[str, ChunkRecord] = {}
        self._chunk_ids: list[str] = []
        self._index: BM25Okapi | None = None

    def rebuild_document_index(self, document_id: str, chunks: list[ChunkRecord]) -> None:
        """Replace one document's source chunks and rebuild the global BM25 corpus."""
        self._chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self._chunks.items()
            if chunk.document_id != document_id
        }
        self._chunks.update({chunk.id: chunk for chunk in chunks})
        self._rebuild()

    def rebuild_all(self, chunks: list[ChunkRecord]) -> None:
        """Replace the complete process-local corpus in one linear rebuild at startup."""
        self._chunks = {chunk.id: chunk for chunk in chunks}
        self._rebuild()

    def has_index(self, document_id: str) -> bool:
        """Report whether the global index currently includes a document's chunks."""
        return any(chunk.document_id == document_id for chunk in self._chunks.values())

    def search(self, query: str, filters: RetrievalFilters, limit: int) -> list[RetrievalCandidate]:
        """Return metadata-filtered BM25 candidates ordered by raw lexical relevance."""
        if self._index is None or limit < 1:
            return []
        query_tokens = self._tokenize(query)
        query_token_set = set(query_tokens)
        raw_scores = self._index.get_scores(query_tokens)
        scored = [
            (self._chunks[chunk_id], float(score))
            for chunk_id, score in zip(self._chunk_ids, raw_scores, strict=True)
            if _matches_filters(self._chunks[chunk_id], filters)
            and (
                float(score) != 0
                or query_token_set.intersection(self._tokenize(self._chunks[chunk_id].text))
            )
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        selected = scored[:limit]
        maximum = max((score for _, score in selected), default=0.0)
        return [
            RetrievalCandidate(
                chunk=chunk,
                source="keyword",
                raw_score=score,
                normalized_score=score / maximum if maximum else 1.0,
            )
            for chunk, score in selected
        ]

    def _rebuild(self) -> None:
        self._chunk_ids = sorted(self._chunks)
        self._index = (
            BM25Okapi([self._tokenize(self._chunks[chunk_id].text) for chunk_id in self._chunk_ids])
            if self._chunk_ids
            else None
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())


def _matches_filters(chunk: ChunkRecord, filters: RetrievalFilters) -> bool:
    return (
        (not filters.document_ids or chunk.document_id in filters.document_ids)
        and (not filters.filenames or chunk.filename in filters.filenames)
        and (not filters.page_numbers or chunk.page_number in filters.page_numbers)
    )
