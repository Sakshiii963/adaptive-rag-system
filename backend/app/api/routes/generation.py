"""Grounded answer endpoint; generation is downstream of the adaptive agent only."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from backend.app.agent.state import AgentState
from backend.app.api.schemas.generation import (
    GenerationRequest,
    GenerationResponse,
    GroundedCitationResponse,
)
from backend.app.api.schemas.reranking import RerankedChunkResponse
from backend.app.core.config import get_settings
from backend.app.dependencies import ApplicationContainer, get_container
from backend.app.domain.entities import RetrievalFilters

router = APIRouter()


@router.post("/answer", response_model=GenerationResponse, status_code=status.HTTP_200_OK)
async def generate_answer(
    request: GenerationRequest,
    container: Annotated[ApplicationContainer, Depends(get_container)],
) -> GenerationResponse:
    """Run adaptive retrieval then generate strictly from its reranked evidence."""
    settings = get_settings()
    filters = RetrievalFilters(
        document_ids=tuple(request.filters.document_ids),
        filenames=tuple(request.filters.filenames),
        page_numbers=tuple(request.filters.page_numbers),
    )
    try:
        agent_state: AgentState = await run_in_threadpool(
            container.adaptive_agent.run,
            request.query,
            filters,
            request.top_k or settings.reranker_top_k,
            request.confidence_threshold
            if request.confidence_threshold is not None
            else settings.agent_confidence_threshold,
            request.max_retries if request.max_retries is not None else settings.agent_max_retries,
        )
        answer = await run_in_threadpool(container.generation_service.generate, agent_state)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return GenerationResponse(
        answer=answer.answer,
        status=answer.status,
        confidence_score=answer.confidence_score,
        evidence_coverage_score=answer.evidence_coverage_score,
        rewritten_query=answer.rewritten_query,
        prompt_version=answer.prompt_version,
        model_name=answer.model_name,
        citations=[
            GroundedCitationResponse(
                marker=citation.marker,
                chunk_id=citation.candidate.candidate.chunk.id,
                document_id=citation.candidate.candidate.chunk.document_id,
                filename=citation.candidate.candidate.chunk.filename,
                page_number=citation.candidate.candidate.chunk.page_number,
            )
            for citation in answer.citations
        ],
        evidence=[
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
            for item in answer.evidence
        ],
    )
