"""Verification-domain records and final report models."""

from dataclasses import dataclass
from typing import Literal

from backend.app.domain.entities import GroundedAnswer, RerankedCandidate


@dataclass(frozen=True, slots=True)
class AtomicClaim:
    """A sentence-level factual unit and the citation markers it declares."""

    claim_id: int
    text: str
    original_text: str
    citation_markers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ClaimVerification:
    """Support decision and per-citation semantic scores for one claim."""

    claim_id: int
    text: str
    citation_markers: tuple[int, ...]
    support_scores: tuple[float, ...]
    supported: bool
    reason: str


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Complete verification audit, including unsupported claims and citation references."""

    claims: tuple[ClaimVerification, ...]
    unsupported_claim_ids: tuple[int, ...]
    unsupported_citations: tuple[int, ...]
    citation_coverage_score: float
    grounding_score: float
    retry_triggered: bool
    retry_count: int


@dataclass(frozen=True, slots=True)
class VerifiedAnswer:
    """Final claim-filtered or repaired answer with an independent verification report."""

    answer: str
    status: Literal["verified", "repaired", "insufficient_evidence"]
    confidence_score: float
    citation_coverage_score: float
    grounding_score: float
    report: VerificationReport
    evidence: tuple[RerankedCandidate, ...]
    rewritten_query: str | None
    prompt_version: str
    model_name: str
    source_answer: GroundedAnswer
