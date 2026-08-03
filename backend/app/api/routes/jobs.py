"""Indexing progress routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.app.api.schemas.jobs import IndexingJobListResponse, IndexingJobResponse
from backend.app.dependencies import ApplicationContainer, get_container

router = APIRouter()


@router.get("/{job_id}", response_model=IndexingJobResponse)
async def get_indexing_job(
    job_id: str,
    container: Annotated[ApplicationContainer, Depends(get_container)],
) -> IndexingJobResponse:
    """Return the live persisted progress of one indexing job."""
    job = container.repository.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Indexing job not found.")
    return _to_response(job)


@router.get("", response_model=IndexingJobListResponse)
async def list_indexing_jobs(
    container: Annotated[ApplicationContainer, Depends(get_container)],
    document_id: Annotated[str | None, Query()] = None,
) -> IndexingJobListResponse:
    """List persisted jobs, optionally constrained to one document."""
    return IndexingJobListResponse(
        jobs=[_to_response(job) for job in container.repository.list_jobs(document_id)]
    )


def _to_response(job) -> IndexingJobResponse:
    return IndexingJobResponse(
        job_id=job.id,
        document_id=job.document_id,
        status=job.status,
        progress=job.progress,
        stage=job.stage,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )
