"""Document acceptance and background indexing application service."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import fitz
from fastapi import UploadFile

from backend.app.core.logging import get_logger
from backend.app.domain.entities import DocumentRecord, IndexingJobRecord
from backend.app.infrastructure.bm25.index import BM25IndexManager
from backend.app.infrastructure.database.sqlite_repository import SQLiteMetadataRepository
from backend.app.infrastructure.embedding.bge_embeddings import EmbeddingProvider
from backend.app.infrastructure.vector.chroma_store import ChromaChunkStore
from backend.app.services.chunking import SemanticChunker

logger = get_logger(__name__)


class UploadTooLargeError(ValueError):
    """Raised when an upload exceeds the configured capacity limit."""


class InvalidPdfError(ValueError):
    """Raised when a submitted file is not a readable PDF."""


class IngestionService:
    """Coordinates durable upload acceptance and idempotent document indexing."""

    def __init__(
        self,
        repository: SQLiteMetadataRepository,
        vector_store: ChromaChunkStore,
        embedding_provider: EmbeddingProvider,
        bm25_manager: BM25IndexManager,
        chunker: SemanticChunker,
        upload_directory: Path,
        max_upload_size_bytes: int,
    ) -> None:
        self.repository = repository
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.bm25_manager = bm25_manager
        self.chunker = chunker
        self.upload_directory = upload_directory
        self.max_upload_size_bytes = max_upload_size_bytes

    async def accept_upload(
        self, upload: UploadFile
    ) -> tuple[DocumentRecord, IndexingJobRecord | None, bool]:
        """Persist one upload and schedule-ready metadata, deduplicating by SHA-256."""
        filename = _safe_filename(upload.filename)
        if not filename.lower().endswith(".pdf"):
            raise InvalidPdfError("Only PDF files are supported.")
        self.upload_directory.mkdir(parents=True, exist_ok=True)
        temporary_path = self.upload_directory / f".{uuid4()}.upload"
        digest = hashlib.sha256()
        total_bytes = 0
        try:
            with temporary_path.open("wb") as target:
                while block := await upload.read(1024 * 1024):
                    total_bytes += len(block)
                    if total_bytes > self.max_upload_size_bytes:
                        raise UploadTooLargeError("The PDF exceeds the configured upload limit.")
                    digest.update(block)
                    target.write(block)
            sha256 = digest.hexdigest()
            duplicate = self.repository.get_document_by_hash(sha256)
            if duplicate:
                temporary_path.unlink(missing_ok=True)
                return duplicate, None, True
            document_id = str(uuid4())
            destination = self.upload_directory / f"{document_id}.pdf"
            temporary_path.replace(destination)
            now = datetime.now(UTC)
            document = DocumentRecord(
                id=document_id,
                sha256=sha256,
                filename=filename,
                storage_path=str(destination),
                upload_timestamp=now,
                status="queued",
                page_count=None,
                chunk_count=0,
                error_message=None,
            )
            job = IndexingJobRecord(
                id=str(uuid4()),
                document_id=document_id,
                status="queued",
                progress=0,
                stage="queued",
                error_message=None,
                created_at=now,
                started_at=None,
                completed_at=None,
            )
            self.repository.create_document(document)
            self.repository.create_job(job)
            return document, job, False
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()

    def index_document(self, document_id: str, job_id: str) -> None:
        """Run a bounded, observable background indexing operation for one document."""
        document = self.repository.get_document(document_id)
        if document is None:
            logger.error(
                "indexing_document_missing", extra={"document_id": document_id, "job_id": job_id}
            )
            return
        self.repository.update_job(job_id, "running", 5, "opening_pdf")
        try:
            pages = self._extract_pages(Path(document.storage_path))
            self.repository.update_job(job_id, "running", 30, "chunking")
            chunks = []
            for page_number, text in pages:
                chunks.extend(
                    self.chunker.chunk_page(
                        document.id, document.filename, page_number, text, document.upload_timestamp
                    )
                )
            if not chunks:
                raise InvalidPdfError("The PDF did not contain extractable text.")
            self.repository.update_job(job_id, "running", 55, "embedding")
            embeddings = self.embedding_provider.embed_documents([chunk.text for chunk in chunks])
            self.repository.update_job(job_id, "running", 75, "persisting_vectors")
            self.vector_store.replace_document(chunks, embeddings)
            self.repository.update_job(job_id, "running", 88, "persisting_metadata")
            self.repository.replace_chunks(document.id, chunks)
            self.bm25_manager.rebuild_document_index(document.id, chunks)
            self.repository.update_document_indexed(document.id, len(pages), len(chunks))
            self.repository.update_job(job_id, "completed", 100, "completed")
            logger.info(
                "indexing_completed",
                extra={"document_id": document_id, "job_id": job_id, "chunk_count": len(chunks)},
            )
        except Exception as exc:
            message = str(exc) or "Document indexing failed."
            self.repository.update_document_failed(document_id, message)
            self.repository.update_job(job_id, "failed", 100, "failed", message)
            logger.exception(
                "indexing_failed", extra={"document_id": document_id, "job_id": job_id}
            )

    @staticmethod
    def _extract_pages(path: Path) -> list[tuple[int, str]]:
        try:
            with fitz.open(path) as pdf:
                return [(page.number + 1, page.get_text("text").strip()) for page in pdf]
        except (fitz.FileDataError, RuntimeError, OSError) as exc:
            raise InvalidPdfError("The uploaded file is not a readable PDF.") from exc


def _safe_filename(filename: str | None) -> str:
    candidate = Path(filename or "document.pdf").name.strip()
    return candidate or "document.pdf"
