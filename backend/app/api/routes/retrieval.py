"""Evidence-only hybrid retrieval endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from backend.app.api.schemas.retrieval import (
    RetrievalLatencyResponse,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunkResponse,
)
from backend.app.core.config import get_settings
from backend.app.dependencies import ApplicationContainer, get_container
from backend.app.domain.entities import RetrievalFilters

router = APIRouter()


@router.post("/search", response_model=RetrievalResponse, status_code=status.HTTP_200_OK)
async def search(
    request: RetrievalRequest,
    container: Annotated[ApplicationContainer, Depends(get_container)],
) -> RetrievalResponse:
    """Run parallel semantic and BM25 retrieval; no LLM, agent, or answer generation is involved."""
    filters = RetrievalFilters(
        document_ids=tuple(request.filters.document_ids),
        filenames=tuple(request.filters.filenames),
        page_numbers=tuple(request.filters.page_numbers),
    )
    try:
        result = await run_in_threadpool(
            container.retrieval_engine.retrieve,
            request.query,
            filters,
            request.top_k or get_settings().retrieval_top_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return RetrievalResponse(
        query=request.query,
        confidence_score=result.confidence_score,
        results=[
            RetrievedChunkResponse(
                chunk_id=candidate.chunk.id,
                document_id=candidate.chunk.document_id,
                filename=candidate.chunk.filename,
                page_number=candidate.chunk.page_number,
                sequence=candidate.chunk.sequence,
                text=candidate.chunk.text,
                semantic_score=candidate.semantic_score,
                keyword_score=candidate.keyword_score,
                rrf_score=candidate.rrf_score,
                normalized_score=candidate.normalized_score,
            )
            for candidate in result.candidates
        ],
        latency=RetrievalLatencyResponse(
            semantic_ms=result.semantic_latency_ms,
            keyword_ms=result.keyword_latency_ms,
            total_ms=result.total_latency_ms,
        ),
    )
