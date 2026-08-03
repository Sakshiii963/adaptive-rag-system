"""ChromaDB persistence adapter for ingestion-time vector writes."""

from pathlib import Path

import chromadb

from backend.app.domain.entities import ChunkRecord, RetrievalCandidate, RetrievalFilters


class ChromaChunkStore:
    """Owns a persistent Chroma collection of citation-ready document chunks."""

    def __init__(self, persist_directory: str, collection_name: str) -> None:
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def replace_document(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        """Replace all vectors for one document so retrying an index job is safe."""
        if len(chunks) != len(embeddings):
            raise ValueError("Every chunk must have exactly one embedding.")
        if not chunks:
            return
        document_id = chunks[0].document_id
        self._collection.delete(where={"document_id": document_id})
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "page_number": chunk.page_number,
                    "chunk_id": chunk.id,
                    "sequence": chunk.sequence,
                    "upload_timestamp": chunk.upload_timestamp.isoformat(),
                }
                for chunk in chunks
            ],
        )

    def count(self) -> int:
        """Return collection cardinality for health and integration verification."""
        return self._collection.count()

    def search(
        self, query_embedding: list[float], filters: RetrievalFilters, limit: int
    ) -> list[RetrievalCandidate]:
        """Perform vector similarity search and map cosine distances to normalized candidates."""
        if limit < 1:
            return []
        collection_count = self._collection.count()
        if collection_count == 0:
            return []
        response = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, collection_count),
            where=_to_chroma_filter(filters),
            include=["documents", "metadatas", "distances"],
        )
        ids = response["ids"][0]
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]
        raw_scores = [max(0.0, min(1.0, 1.0 - (float(distance) / 2.0))) for distance in distances]
        return [
            RetrievalCandidate(
                chunk=ChunkRecord(
                    id=chunk_id,
                    document_id=str(metadata["document_id"]),
                    filename=str(metadata["filename"]),
                    page_number=int(metadata["page_number"]),
                    sequence=int(metadata["sequence"]),
                    text=str(document),
                    upload_timestamp=_parse_timestamp(str(metadata["upload_timestamp"])),
                ),
                source="semantic",
                raw_score=score,
                normalized_score=score,
            )
            for chunk_id, document, metadata, score in zip(
                ids, documents, metadatas, raw_scores, strict=True
            )
        ]


def _to_chroma_filter(filters: RetrievalFilters) -> dict[str, object] | None:
    clauses: list[dict[str, object]] = []
    if filters.document_ids:
        clauses.append({"document_id": _filter_value(filters.document_ids)})
    if filters.filenames:
        clauses.append({"filename": _filter_value(filters.filenames)})
    if filters.page_numbers:
        clauses.append({"page_number": _filter_value(filters.page_numbers)})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _filter_value(values: tuple[str, ...] | tuple[int, ...]) -> object:
    return values[0] if len(values) == 1 else {"$in": list(values)}


def _parse_timestamp(value: str):
    from datetime import datetime

    return datetime.fromisoformat(value)
