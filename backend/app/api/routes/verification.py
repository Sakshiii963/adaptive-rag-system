"""Verified grounded-answer endpoint independent from generation implementation."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from backend.app.agent.state import AgentState
from backend.app.api.schemas.reranking import RerankedChunkResponse
from backend.app.api.schemas.verification import (
    ClaimVerificationResponse,
    VerificationReportResponse,
    VerificationRequest,
    VerificationResponse,
)
from backend.app.core.config import get_settings
from backend.app.dependencies import ApplicationContainer, get_container
from backend.app.domain.entities import RetrievalFilters

router = APIRouter()


@router.post("/answer", response_model=VerificationResponse, status_code=status.HTTP_200_OK)
async def verify_answer(
    request: VerificationRequest,
    container: Annotated[ApplicationContainer, Depends(get_container)],
) -> VerificationResponse:
    """Generate from adaptive evidence, verify every claim, and repair weak evidence when possible."""
    settings = get_settings()
    filters = RetrievalFilters(
        document_ids=tuple(request.filters.document_ids),
        filenames=tuple(request.filters.filenames),
        page_numbers=tuple(request.filters.page_numbers),
    )

    def retry(target_query: str) -> AgentState:
        return container.adaptive_agent.run(
            target_query,
            filters,
            request.top_k or settings.reranker_top_k,
            request.confidence_threshold
            if request.confidence_threshold is not None
            else settings.agent_confidence_threshold,
            request.max_retries if request.max_retries is not None else settings.agent_max_retries,
        )

    try:
        initial_state: AgentState = await run_in_threadpool(retry, request.query)
        initial_answer = await run_in_threadpool(
            container.generation_service.generate, initial_state
        )
        verified = await run_in_threadpool(
            container.verification_service.verify,
            initial_answer,
            retry,
            container.generation_service.generate,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    report = verified.report
    return VerificationResponse(
        answer=verified.answer,
        status=verified.status,
        confidence_score=verified.confidence_score,
        citation_coverage_score=verified.citation_coverage_score,
        grounding_score=verified.grounding_score,
        rewritten_query=verified.rewritten_query,
        report=VerificationReportResponse(
            claims=[
                ClaimVerificationResponse(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    citation_markers=list(claim.citation_markers),
                    support_scores=list(claim.support_scores),
                    supported=claim.supported,
                    reason=claim.reason,
                )
                for claim in report.claims
            ],
            unsupported_claim_ids=list(report.unsupported_claim_ids),
            unsupported_citations=list(report.unsupported_citations),
            citation_coverage_score=report.citation_coverage_score,
            grounding_score=report.grounding_score,
            retry_triggered=report.retry_triggered,
            retry_count=report.retry_count,
        ),
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
            for item in verified.evidence
        ],
        trace=list(initial_state.get("trace", [])),
    )
