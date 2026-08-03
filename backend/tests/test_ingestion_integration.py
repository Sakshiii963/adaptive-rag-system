"""Integration tests for SQLite, PyMuPDF extraction, BM25 construction, and Chroma persistence."""

from datetime import UTC, datetime
from pathlib import Path

import fitz

from backend.app.domain.entities import DocumentRecord, IndexingJobRecord
from backend.app.infrastructure.bm25.index import BM25IndexManager
from backend.app.infrastructure.database.sqlite_repository import SQLiteMetadataRepository
from backend.app.infrastructure.vector.chroma_store import ChromaChunkStore
from backend.app.services.chunking import SemanticChunker
from backend.app.services.ingestion import IngestionService


class FakeEmbeddingProvider:
    """Deterministic vectors keep integration tests local and model-download-free."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]


def test_indexing_persists_metadata_vectors_and_bm25_index(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    pdf = fitz.open()
    first = pdf.new_page()
    first.insert_text(
        (72, 72), "Adaptive RAG improves evidence quality. It preserves page provenance."
    )
    second = pdf.new_page()
    second.insert_text((72, 72), "Second page discusses metadata persistence and document hashes.")
    pdf.save(pdf_path)
    pdf.close()

    repository = SQLiteMetadataRepository(tmp_path / "metadata.sqlite3")
    repository.initialize()
    vector_store = ChromaChunkStore(str(tmp_path / "chroma"), "test_chunks")
    bm25 = BM25IndexManager()
    document = DocumentRecord(
        id="doc-1",
        sha256="a" * 64,
        filename="source.pdf",
        storage_path=str(pdf_path),
        upload_timestamp=datetime.now(UTC),
        status="queued",
        page_count=None,
        chunk_count=0,
        error_message=None,
    )
    job = IndexingJobRecord(
        id="job-1",
        document_id=document.id,
        status="queued",
        progress=0,
        stage="queued",
        error_message=None,
        created_at=datetime.now(UTC),
        started_at=None,
        completed_at=None,
    )
    repository.create_document(document)
    repository.create_job(job)
    service = IngestionService(
        repository=repository,
        vector_store=vector_store,
        embedding_provider=FakeEmbeddingProvider(),
        bm25_manager=bm25,
        chunker=SemanticChunker(120, 20),
        upload_directory=tmp_path / "uploads",
        max_upload_size_bytes=1024 * 1024,
    )

    service.index_document(document.id, job.id)

    indexed_document = repository.get_document(document.id)
    indexed_job = repository.get_job(job.id)
    assert indexed_document is not None
    assert indexed_document.status == "indexed"
    assert indexed_document.page_count == 2
    assert indexed_document.chunk_count >= 2
    assert indexed_job is not None
    assert indexed_job.status == "completed"
    assert vector_store.count() == indexed_document.chunk_count
    assert bm25.has_index(document.id)

    reopened_store = ChromaChunkStore(str(tmp_path / "chroma"), "test_chunks")
    assert reopened_store.count() == indexed_document.chunk_count


def test_duplicate_hash_returns_existing_document(tmp_path: Path) -> None:
    repository = SQLiteMetadataRepository(tmp_path / "metadata.sqlite3")
    repository.initialize()
    document = DocumentRecord(
        id="doc-1",
        sha256="b" * 64,
        filename="first.pdf",
        storage_path="/tmp/first.pdf",
        upload_timestamp=datetime.now(UTC),
        status="queued",
        page_count=None,
        chunk_count=0,
        error_message=None,
    )
    repository.create_document(document)

    duplicate = repository.get_document_by_hash("b" * 64)

    assert duplicate is not None
    assert duplicate.id == document.id
