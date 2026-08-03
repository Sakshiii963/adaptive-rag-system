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


@dataclass(frozen=True, slots=True)
class RetrievalFilters:
    """Optional source-provenance constraints applied consistently to both retrieval channels."""

    document_ids: tuple[str, ...] = ()
    filenames: tuple[str, ...] = ()
    page_numbers: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """A normalized candidate from one retrieval source."""

    chunk: ChunkRecord
    source: Literal["semantic", "keyword"]
    raw_score: float
    normalized_score: float


@dataclass(frozen=True, slots=True)
class HybridRetrievalCandidate:
    """A de-duplicated candidate with source traces and fused reciprocal-rank score."""

    chunk: ChunkRecord
    semantic_score: float | None
    keyword_score: float | None
    rrf_score: float
    normalized_score: float


@dataclass(frozen=True, slots=True)
class HybridRetrievalResult:
    """Complete retrieval result including execution telemetry and confidence."""

    candidates: tuple[HybridRetrievalCandidate, ...]
    confidence_score: float
    semantic_latency_ms: float
    keyword_latency_ms: float
    total_latency_ms: float
