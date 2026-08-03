"""Dependency wiring at the application boundary."""

from dataclasses import dataclass
from pathlib import Path

from fastapi import Request

from backend.app.core.config import Settings
from backend.app.infrastructure.bm25.index import BM25IndexManager
from backend.app.infrastructure.database.sqlite_repository import SQLiteMetadataRepository
from backend.app.infrastructure.embedding.bge_embeddings import BGEEmbeddingProvider
from backend.app.infrastructure.vector.chroma_store import ChromaChunkStore
from backend.app.services.chunking import SemanticChunker
from backend.app.services.ingestion import IngestionService


@dataclass(slots=True)
class ApplicationContainer:
    """Explicit composition root; API routes depend on services, never concrete setup details."""

    repository: SQLiteMetadataRepository
    vector_store: ChromaChunkStore
    bm25_manager: BM25IndexManager
    ingestion_service: IngestionService

    @classmethod
    def create(cls, settings: Settings) -> "ApplicationContainer":
        """Construct long-lived adapters without loading the embedding model prematurely."""
        repository = SQLiteMetadataRepository(settings.metadata_database_path)
        vector_store = ChromaChunkStore(
            settings.chroma_persist_directory, settings.chroma_collection_name
        )
        bm25_manager = BM25IndexManager()
        ingestion_service = IngestionService(
            repository=repository,
            vector_store=vector_store,
            embedding_provider=BGEEmbeddingProvider(),
            bm25_manager=bm25_manager,
            chunker=SemanticChunker(settings.chunk_size, settings.chunk_overlap),
            upload_directory=Path(settings.upload_directory),
            max_upload_size_bytes=settings.max_upload_size_mb * 1024 * 1024,
        )
        return cls(
            repository=repository,
            vector_store=vector_store,
            bm25_manager=bm25_manager,
            ingestion_service=ingestion_service,
        )

    def initialize(self) -> None:
        """Initialize durable metadata before requests and jobs are accepted."""
        self.repository.initialize()
        for document_id in self.repository.list_indexed_document_ids():
            self.bm25_manager.rebuild_document_index(
                document_id, self.repository.list_chunks(document_id)
            )

    def close(self) -> None:
        """Release resources; current adapters have no explicit close action."""


def get_container(request: Request) -> ApplicationContainer:
    """Resolve the initialized application container for API dependencies."""
    return request.app.state.container
