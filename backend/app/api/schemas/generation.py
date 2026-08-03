"""Grounded generation API contracts."""

from typing import Literal

from pydantic import BaseModel, Field

from backend.app.api.schemas.agent import AgentRequest
from backend.app.api.schemas.reranking import RerankedChunkResponse


class GenerationRequest(AgentRequest):
    """Adaptive retrieval controls for one grounded answer request."""


class GroundedCitationResponse(BaseModel):
    """Inline marker mapped to preserved source provenance."""

    marker: int
    chunk_id: str
    document_id: str
    filename: str
    page_number: int


class GenerationResponse(BaseModel):
    """Grounded answer with explicit evidence and safety metrics."""

    answer: str
    status: Literal["answer", "insufficient_evidence"]
    confidence_score: float = Field(ge=0, le=1)
    evidence_coverage_score: float = Field(ge=0, le=1)
    rewritten_query: str | None
    prompt_version: str
    model_name: str
    citations: list[GroundedCitationResponse]
    evidence: list[RerankedChunkResponse]
