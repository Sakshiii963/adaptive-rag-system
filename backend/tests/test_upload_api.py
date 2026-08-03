"""API integration coverage for multipart multi-file ingestion and job polling."""

from pathlib import Path

import fitz
from fastapi.testclient import TestClient

from backend.app.dependencies import ApplicationContainer
from backend.app.infrastructure.bm25.index import BM25IndexManager
from backend.app.infrastructure.database.sqlite_repository import SQLiteMetadataRepository
from backend.app.infrastructure.reranker.cross_encoder import LocalCrossEncoder
from backend.app.infrastructure.vector.chroma_store import ChromaChunkStore
from backend.app.main import create_app
from backend.app.services.chunking import SemanticChunker
from backend.app.services.ingestion import IngestionService
from backend.app.services.reranking import RerankingService
from backend.app.services.retrieval import HybridRetrievalEngine


class FakeEmbeddingProvider:
    """Keeps the API integration test independent of model downloads."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]


def test_multi_pdf_upload_starts_persisted_jobs(tmp_path: Path) -> None:
    repository = SQLiteMetadataRepository(tmp_path / "metadata.sqlite3")
    repository.initialize()
    vector_store = ChromaChunkStore(str(tmp_path / "chroma"), "upload_chunks")
    ingestion = IngestionService(
        repository=repository,
        vector_store=vector_store,
        embedding_provider=FakeEmbeddingProvider(),
        bm25_manager=BM25IndexManager(),
        chunker=SemanticChunker(300, 50),
        upload_directory=tmp_path / "uploads",
        max_upload_size_bytes=1024 * 1024,
    )
    container = ApplicationContainer(
        repository=repository,
        vector_store=vector_store,
        bm25_manager=BM25IndexManager(),
        ingestion_service=ingestion,
        retrieval_engine=HybridRetrievalEngine(
            vector_store=vector_store,
            embedding_provider=ingestion.embedding_provider,
            bm25_manager=ingestion.bm25_manager,
            rrf_constant=60,
            candidate_multiplier=4,
        ),
        reranking_service=RerankingService(
            provider=LocalCrossEncoder(), model_name="test", batch_size=2
        ),
    )
    app = create_app()
    with TestClient(app) as client:
        app.state.container = container
        response = client.post(
            "/api/v1/documents/upload",
            files=[
                ("files", ("first.pdf", _pdf_bytes("First PDF source text."), "application/pdf")),
                ("files", ("second.pdf", _pdf_bytes("Second PDF source text."), "application/pdf")),
            ],
        )

        assert response.status_code == 202
        accepted = response.json()["documents"]
        assert len(accepted) == 2
        assert all(item["job_id"] for item in accepted)
        assert all(item["duplicate"] is False for item in accepted)

        job_response = client.get(f"/api/v1/jobs/{accepted[0]['job_id']}")
        assert job_response.status_code == 200
        assert job_response.json()["status"] == "completed"
        assert job_response.json()["progress"] == 100
        assert vector_store.count() == 2


def _pdf_bytes(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content
