"""Dependency wiring at the application boundary."""

from dataclasses import dataclass
from pathlib import Path

from fastapi import Request

from backend.app.agent.graph import AdaptiveRetrievalAgent
from backend.app.core.config import Settings
from backend.app.generation.context import ContextWindowManager
from backend.app.generation.prompt import GroundedPromptBuilder
from backend.app.generation.service import GroundedGenerationService
from backend.app.infrastructure.bm25.index import BM25IndexManager
from backend.app.infrastructure.database.sqlite_repository import SQLiteMetadataRepository
from backend.app.infrastructure.embedding.bge_embeddings import BGEEmbeddingProvider
from backend.app.infrastructure.llm.ollama import OllamaGenerationProvider
from backend.app.infrastructure.reranker.cross_encoder import LocalCrossEncoder
from backend.app.infrastructure.vector.chroma_store import ChromaChunkStore
from backend.app.services.chunking import SemanticChunker
from backend.app.services.ingestion import IngestionService
from backend.app.services.reranking import RerankingService
from backend.app.services.retrieval import HybridRetrievalEngine
from backend.app.verification.service import CitationVerificationService
from backend.app.verification.support import SemanticSupportVerifier


@dataclass(slots=True)
class ApplicationContainer:
    """Explicit composition root; API routes depend on services, never concrete setup details."""

    repository: SQLiteMetadataRepository
    vector_store: ChromaChunkStore
    bm25_manager: BM25IndexManager
    ingestion_service: IngestionService
    retrieval_engine: HybridRetrievalEngine
    reranking_service: RerankingService
    adaptive_agent: AdaptiveRetrievalAgent
    generation_service: GroundedGenerationService
    verification_service: CitationVerificationService

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
        retrieval_engine = HybridRetrievalEngine(
            vector_store=vector_store,
            embedding_provider=ingestion_service.embedding_provider,
            bm25_manager=bm25_manager,
            rrf_constant=settings.rrf_constant,
            candidate_multiplier=settings.retrieval_candidate_multiplier,
        )
        reranking_service = RerankingService(
            provider=LocalCrossEncoder(settings.reranker_model_name),
            model_name=settings.reranker_model_name,
            batch_size=settings.reranker_batch_size,
        )
        adaptive_agent = AdaptiveRetrievalAgent(
            retrieval_engine=retrieval_engine,
            reranking_service=reranking_service,
        )
        generation_service = GroundedGenerationService(
            provider=OllamaGenerationProvider(
                settings.ollama_base_url, settings.ollama_model, settings.ollama_timeout_seconds
            ),
            model_name=settings.ollama_model,
            prompt_builder=GroundedPromptBuilder(settings.generation_prompt_version),
            context_manager=ContextWindowManager(settings.generation_context_max_chars),
            max_output_tokens=settings.generation_max_output_tokens,
        )
        verification_service = CitationVerificationService(
            support_verifier=SemanticSupportVerifier(
                scorer=reranking_service.provider,
                threshold=settings.verification_support_threshold,
                batch_size=settings.verification_batch_size,
            ),
            min_coverage=settings.verification_min_coverage,
            max_retries=settings.verification_max_retries,
        )
        return cls(
            repository=repository,
            vector_store=vector_store,
            bm25_manager=bm25_manager,
            ingestion_service=ingestion_service,
            retrieval_engine=retrieval_engine,
            reranking_service=reranking_service,
            adaptive_agent=adaptive_agent,
            generation_service=generation_service,
            verification_service=verification_service,
        )

    def initialize(self) -> None:
        """Initialize durable metadata before requests and jobs are accepted."""
        self.repository.initialize()
        indexed_chunks = [
            chunk
            for document_id in self.repository.list_indexed_document_ids()
            for chunk in self.repository.list_chunks(document_id)
        ]
        self.bm25_manager.rebuild_all(indexed_chunks)

    def close(self) -> None:
        """Release resources; current adapters have no explicit close action."""


def get_container(request: Request) -> ApplicationContainer:
    """Resolve the initialized application container for API dependencies."""
    return request.app.state.container
