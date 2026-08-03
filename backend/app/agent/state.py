"""Typed state carried through the adaptive retrieval graph."""

from typing import Any, Literal, TypedDict

from backend.app.domain.entities import (
    HybridRetrievalResult,
    RerankingResult,
    RetrievalFilters,
)


class TraceStep(TypedDict):
    """Serializable structured record for one graph-node execution."""

    node: str
    attempt: int
    timestamp: str
    duration_ms: float
    details: dict[str, Any]


class AgentState(TypedDict, total=False):
    """Explicit LangGraph state; no LLM messages or generated answers are stored."""

    original_query: str
    current_query: str
    rewritten_query: str | None
    attempt: int
    max_retries: int
    top_k: int
    confidence_threshold: float
    filters: RetrievalFilters
    retrieval_result: HybridRetrievalResult
    reranking_result: RerankingResult
    confidence: float
    status: Literal["running", "evidence", "insufficient_evidence"]
    stop_reason: str | None
    seen_queries: list[str]
    trace: list[TraceStep]
    reasoning_steps: list[str]
    next_route: Literal["retry_retrieval", "insufficient_evidence"]
