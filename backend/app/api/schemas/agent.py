"""Adaptive retrieval-agent request and evidence-only response contracts."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.app.api.schemas.reranking import RerankedChunkResponse
from backend.app.api.schemas.retrieval import RetrievalFilterRequest


class AgentRequest(BaseModel):
    """Controls for one bounded adaptive retrieval execution."""

    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    confidence_threshold: float | None = Field(default=None, ge=0, le=1)
    max_retries: int | None = Field(default=None, ge=0, le=10)
    filters: RetrievalFilterRequest = Field(default_factory=RetrievalFilterRequest)


class AgentResponse(BaseModel):
    """The agent's trace and reranked evidence, with no generated answer field."""

    query: str
    rewritten_query: str | None
    status: Literal["evidence", "insufficient_evidence"]
    confidence: float = Field(ge=0, le=1)
    reasoning_steps: list[str]
    retrieval_trace: list[dict[str, Any]]
    reranked_evidence: list[RerankedChunkResponse]
