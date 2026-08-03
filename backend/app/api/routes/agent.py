"""Adaptive LangGraph retrieval-agent endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from backend.app.agent.state import AgentState
from backend.app.api.schemas.agent import AgentRequest, AgentResponse
from backend.app.api.schemas.reranking import RerankedChunkResponse
from backend.app.core.config import get_settings
from backend.app.dependencies import ApplicationContainer, get_container
from backend.app.domain.entities import RetrievalFilters

router = APIRouter()


@router.post("/run", response_model=AgentResponse, status_code=status.HTTP_200_OK)
async def run_agent(
    request: AgentRequest,
    container: Annotated[ApplicationContainer, Depends(get_container)],
) -> AgentResponse:
    """Run adaptive retrieval only; no LLM response generation occurs."""
    settings = get_settings()
    filters = RetrievalFilters(
        document_ids=tuple(request.filters.document_ids),
        filenames=tuple(request.filters.filenames),
        page_numbers=tuple(request.filters.page_numbers),
    )
    try:
        state: AgentState = await run_in_threadpool(
            container.adaptive_agent.run,
            request.query,
            filters,
            request.top_k or settings.reranker_top_k,
            request.confidence_threshold
            if request.confidence_threshold is not None
            else settings.agent_confidence_threshold,
            request.max_retries if request.max_retries is not None else settings.agent_max_retries,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    reranking_result = state.get("reranking_result")
    evidence = []
    if reranking_result:
        evidence = [
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
        ]
    return AgentResponse(
        query=state["original_query"],
        rewritten_query=state.get("rewritten_query"),
        status="evidence" if state.get("status") == "evidence" else "insufficient_evidence",
        confidence=state.get("confidence", 0.0),
        reasoning_steps=state.get("reasoning_steps", []),
        retrieval_trace=state.get("trace", []),
        reranked_evidence=evidence,
    )
