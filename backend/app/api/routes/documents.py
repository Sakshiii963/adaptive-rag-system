"""Multi-file PDF upload and document metadata routes."""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status

from backend.app.api.schemas.documents import (
    DocumentResponse,
    MultiUploadResponse,
    UploadedDocumentResponse,
    UploadFailure,
)
from backend.app.dependencies import ApplicationContainer, get_container
from backend.app.services.ingestion import InvalidPdfError, UploadTooLargeError

router = APIRouter()


@router.post("/upload", response_model=MultiUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile], File(description="One or more PDF files")],
    container: Annotated[ApplicationContainer, Depends(get_container)],
) -> MultiUploadResponse:
    """Accept multiple files from a file picker or browser drag-and-drop form submission."""
    accepted: list[UploadedDocumentResponse] = []
    rejected: list[UploadFailure] = []
    for file in files:
        try:
            document, job, duplicate = await container.ingestion_service.accept_upload(file)
        except UploadTooLargeError as exc:
            rejected.append(UploadFailure(filename=file.filename or "unknown", message=str(exc)))
            continue
        except InvalidPdfError as exc:
            rejected.append(UploadFailure(filename=file.filename or "unknown", message=str(exc)))
            continue
        if job:
            background_tasks.add_task(
                container.ingestion_service.index_document, document.id, job.id
            )
        accepted.append(
            UploadedDocumentResponse(
                document_id=document.id,
                filename=document.filename,
                status=document.status,
                duplicate=duplicate,
                job_id=job.id if job else None,
                upload_timestamp=document.upload_timestamp,
            )
        )
    return MultiUploadResponse(documents=accepted, rejected=rejected)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    container: Annotated[ApplicationContainer, Depends(get_container)],
) -> DocumentResponse:
    """Return document metadata and final indexing state."""
    document = container.repository.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return DocumentResponse(
        document_id=document.id,
        sha256=document.sha256,
        filename=document.filename,
        status=document.status,
        page_count=document.page_count,
        chunk_count=document.chunk_count,
        upload_timestamp=document.upload_timestamp,
        error_message=document.error_message,
    )
