"""Document-ingestion HTTP contracts."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class UploadedDocumentResponse(BaseModel):
    """Result for one file accepted by the multi-file upload endpoint."""

    document_id: str
    filename: str
    status: Literal["queued", "indexing", "indexed", "failed"]
    duplicate: bool
    job_id: str | None
    upload_timestamp: datetime


class MultiUploadResponse(BaseModel):
    """Batch response for browser file-picker and drag-and-drop uploads."""

    documents: list[UploadedDocumentResponse]
    rejected: list["UploadFailure"] = Field(default_factory=list)


class DocumentResponse(BaseModel):
    """Persisted document metadata without exposing its local storage path."""

    document_id: str
    sha256: str
    filename: str
    status: Literal["queued", "indexing", "indexed", "failed"]
    page_count: int | None
    chunk_count: int
    upload_timestamp: datetime
    error_message: str | None = None


class UploadFailure(BaseModel):
    """Per-file failure returned without abandoning other files in a batch."""

    filename: str
    message: str
