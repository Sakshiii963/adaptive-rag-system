"""SQLite repository for document, job, and chunk metadata."""

import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from backend.app.domain.entities import ChunkRecord, DocumentRecord, IndexingJobRecord


class SQLiteMetadataRepository:
    """Owns SQLite schema initialization and small transactional metadata operations."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        """Create durable tables and indexes if this is the first application startup."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    upload_timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    page_count INTEGER,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT
                );
                CREATE TABLE IF NOT EXISTS indexing_jobs (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id),
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id),
                    filename TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    sequence INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    upload_timestamp TEXT NOT NULL,
                    UNIQUE(document_id, page_number, sequence)
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_document_id ON indexing_jobs(document_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
                """
            )

    def create_document(self, document: DocumentRecord) -> None:
        """Insert a newly accepted, unique document."""
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO documents
                (id, sha256, filename, storage_path, upload_timestamp, status, page_count, chunk_count,
                 error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    document.id,
                    document.sha256,
                    document.filename,
                    document.storage_path,
                    _serialize_datetime(document.upload_timestamp),
                    document.status,
                    document.page_count,
                    document.chunk_count,
                    document.error_message,
                ),
            )

    def get_document_by_hash(self, sha256: str) -> DocumentRecord | None:
        """Return an existing document with the same content hash, if any."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return _to_document(row) if row else None

    def get_document(self, document_id: str) -> DocumentRecord | None:
        """Load one document record."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return _to_document(row) if row else None

    def list_indexed_document_ids(self) -> list[str]:
        """Return documents whose durable chunks should be rebuilt into process-local BM25 indexes."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id FROM documents WHERE status = 'indexed'"
            ).fetchall()
        return [row["id"] for row in rows]

    def update_document_indexed(self, document_id: str, page_count: int, chunk_count: int) -> None:
        """Mark a document fully indexed only after all stores have been written."""
        with self._connection() as connection:
            connection.execute(
                """UPDATE documents SET status = 'indexed', page_count = ?, chunk_count = ?,
                error_message = NULL WHERE id = ?""",
                (page_count, chunk_count, document_id),
            )

    def update_document_failed(self, document_id: str, message: str) -> None:
        """Record a bounded failure message for operator inspection."""
        with self._connection() as connection:
            connection.execute(
                "UPDATE documents SET status = 'failed', error_message = ? WHERE id = ?",
                (message[:1000], document_id),
            )

    def create_job(self, job: IndexingJobRecord) -> None:
        """Persist an initially queued indexing job."""
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO indexing_jobs
                (id, document_id, status, progress, stage, error_message, created_at, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job.id,
                    job.document_id,
                    job.status,
                    job.progress,
                    job.stage,
                    job.error_message,
                    _serialize_datetime(job.created_at),
                    _serialize_datetime(job.started_at) if job.started_at else None,
                    _serialize_datetime(job.completed_at) if job.completed_at else None,
                ),
            )

    def get_job(self, job_id: str) -> IndexingJobRecord | None:
        """Load one job record."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM indexing_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return _to_job(row) if row else None

    def list_jobs(self, document_id: str | None = None) -> list[IndexingJobRecord]:
        """List jobs, optionally constrained to a document."""
        query = "SELECT * FROM indexing_jobs"
        values: tuple[str, ...] = ()
        if document_id:
            query += " WHERE document_id = ?"
            values = (document_id,)
        query += " ORDER BY created_at DESC"
        with self._connection() as connection:
            rows = connection.execute(query, values).fetchall()
        return [_to_job(row) for row in rows]

    def update_job(
        self, job_id: str, status: str, progress: int, stage: str, error: str | None = None
    ) -> None:
        """Advance persisted job status, adding lifecycle timestamps when applicable."""
        now = _serialize_datetime(datetime.now(UTC))
        started_at = now if status == "running" else None
        completed_at = now if status in {"completed", "failed"} else None
        with self._connection() as connection:
            connection.execute(
                """UPDATE indexing_jobs SET status = ?, progress = ?, stage = ?, error_message = ?,
                started_at = COALESCE(started_at, ?), completed_at = COALESCE(?, completed_at)
                WHERE id = ?""",
                (
                    status,
                    progress,
                    stage,
                    error[:1000] if error else None,
                    started_at,
                    completed_at,
                    job_id,
                ),
            )

    def replace_chunks(self, document_id: str, chunks: Iterable[ChunkRecord]) -> None:
        """Replace a document's chunks atomically, making indexing retries idempotent."""
        rows = [
            (
                chunk.id,
                chunk.document_id,
                chunk.filename,
                chunk.page_number,
                chunk.sequence,
                chunk.text,
                _serialize_datetime(chunk.upload_timestamp),
            )
            for chunk in chunks
        ]
        with self._connection() as connection:
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            connection.executemany(
                """INSERT INTO chunks
                (id, document_id, filename, page_number, sequence, text, upload_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

    def list_chunks(self, document_id: str) -> list[ChunkRecord]:
        """Load durable chunks in source order for index rebuilding and future retrieval."""
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT * FROM chunks WHERE document_id = ?
                ORDER BY page_number ASC, sequence ASC""",
                (document_id,),
            ).fetchall()
        return [_to_chunk(row) for row in rows]

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection


def _serialize_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _to_document(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"],
        sha256=row["sha256"],
        filename=row["filename"],
        storage_path=row["storage_path"],
        upload_timestamp=datetime.fromisoformat(row["upload_timestamp"]),
        status=row["status"],
        page_count=row["page_count"],
        chunk_count=row["chunk_count"],
        error_message=row["error_message"],
    )


def _to_job(row: sqlite3.Row) -> IndexingJobRecord:
    return IndexingJobRecord(
        id=row["id"],
        document_id=row["document_id"],
        status=row["status"],
        progress=row["progress"],
        stage=row["stage"],
        error_message=row["error_message"],
        created_at=datetime.fromisoformat(row["created_at"]),
        started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
    )


def _to_chunk(row: sqlite3.Row) -> ChunkRecord:
    return ChunkRecord(
        id=row["id"],
        document_id=row["document_id"],
        filename=row["filename"],
        page_number=row["page_number"],
        sequence=row["sequence"],
        text=row["text"],
        upload_timestamp=datetime.fromisoformat(row["upload_timestamp"]),
    )
