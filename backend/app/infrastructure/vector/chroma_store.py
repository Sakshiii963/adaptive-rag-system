"""ChromaDB persistence adapter for ingestion-time vector writes."""

from pathlib import Path

import chromadb

from backend.app.domain.entities import ChunkRecord


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
