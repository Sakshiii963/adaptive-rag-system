"""Domain records used across ingestion infrastructure and services."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

DocumentStatus = Literal["queued", "indexing", "indexed", "failed"]
JobStatus = Literal["queued", "running", "completed", "failed"]


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """Persisted uploaded-document metadata."""

    id: str
    sha256: str
    filename: str
    storage_path: str
    upload_timestamp: datetime
    status: DocumentStatus
    page_count: int | None
    chunk_count: int
    error_message: str | None


@dataclass(frozen=True, slots=True)
class IndexingJobRecord:
    """Persisted, observable indexing-job state."""

    id: str
    document_id: str
    status: JobStatus
    progress: int
    stage: str
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """A citation-ready text fragment belonging to one PDF page."""

    id: str
    document_id: str
    filename: str
    page_number: int
    sequence: int
    text: str
    upload_timestamp: datetime
