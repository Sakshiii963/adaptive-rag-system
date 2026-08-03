"""Indexing-job HTTP contracts."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class IndexingJobResponse(BaseModel):
    """Polling-safe view of a persisted background job."""

    job_id: str
    document_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: int = Field(ge=0, le=100)
    stage: str
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class IndexingJobListResponse(BaseModel):
    """List of jobs, optionally filtered by a document identifier."""

    jobs: list[IndexingJobResponse]
