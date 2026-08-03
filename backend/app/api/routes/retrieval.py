"""Evidence-only hybrid retrieval endpoint."""

from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from backend.app.api.schemas.reranking import (
    RerankedChunkResponse,
    RerankingLatencyResponse,
    RerankingRequest,
    RerankingResponse,
)
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


@router.post("/rerank", response_model=RerankingResponse, status_code=status.HTTP_200_OK)
async def rerank(
    request: RerankingRequest,
    container: Annotated[ApplicationContainer, Depends(get_container)],
) -> RerankingResponse:
    """Retrieve hybrid evidence then reorder it with the local cross-encoder only."""
    filters = RetrievalFilters(
        document_ids=tuple(request.filters.document_ids),
        filenames=tuple(request.filters.filenames),
        page_numbers=tuple(request.filters.page_numbers),
    )
    retrieval_top_k = request.retrieval_top_k or get_settings().retrieval_top_k
    rerank_top_k = request.top_k or get_settings().reranker_top_k
    started_at = perf_counter()
    try:
        retrieval_result = await run_in_threadpool(
            container.retrieval_engine.retrieve, request.query, filters, retrieval_top_k
        )
        reranking_result = await run_in_threadpool(
            container.reranking_service.rerank, request.query, retrieval_result, rerank_top_k
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    total_ms = round((perf_counter() - started_at) * 1000, 2)
    return RerankingResponse(
        query=request.query,
        confidence_score=reranking_result.confidence_score,
        model_name=reranking_result.model_name,
        batch_size=reranking_result.batch_size,
        results=[
            RerankedChunkResponse(
                rank=item.rank,
                chunk_id=item.candidate.chunk.id,
                document_id=item.candidate.chunk.document_id,
                filename=item.candidate.chunk.filename,
                page_number=item.candidate.chunk.page_number,
                sequence=item.candidate.chunk.sequence,
                text=item.candidate.chunk.text,
                semantic_score=item.candidate.semantic_score,
                keyword_score=item.candidate.keyword_score,
                rrf_score=item.candidate.rrf_score,
                retrieval_normalized_score=item.candidate.normalized_score,
                reranker_score=item.reranker_score,
                normalized_reranker_score=item.normalized_reranker_score,
            )
            for item in reranking_result.candidates
        ],
        latency=RerankingLatencyResponse(
            retrieval_ms=retrieval_result.total_latency_ms,
            reranking_ms=reranking_result.latency_ms,
            total_ms=total_ms,
        ),
    )


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
