"""Citation-verification API contracts."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.app.api.schemas.agent import AgentRequest
from backend.app.api.schemas.reranking import RerankedChunkResponse


class VerificationRequest(AgentRequest):
    """Adaptive retrieval controls for one verified grounded answer."""


class ClaimVerificationResponse(BaseModel):
    """Claim-level support result and semantic scores."""

    claim_id: int
    text: str
    citation_markers: list[int]
    support_scores: list[float]
    supported: bool
    reason: str


class VerificationReportResponse(BaseModel):
    """Auditable citation and grounding report."""

    claims: list[ClaimVerificationResponse]
    unsupported_claim_ids: list[int]
    unsupported_citations: list[int]
    citation_coverage_score: float
    grounding_score: float
    retry_triggered: bool
    retry_count: int


class VerificationResponse(BaseModel):
    """Final verified answer and preserved evidence."""

    answer: str
    status: Literal["verified", "repaired", "insufficient_evidence"]
    confidence_score: float = Field(ge=0, le=1)
    citation_coverage_score: float = Field(ge=0, le=1)
    grounding_score: float = Field(ge=0, le=1)
    rewritten_query: str | None
    report: VerificationReportResponse
    evidence: list[RerankedChunkResponse]
    trace: list[dict[str, Any]]
