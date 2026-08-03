"""Independent citation verification, grounding validation, and evidence repair workflow."""

from collections.abc import Callable

from backend.app.agent.state import AgentState
from backend.app.domain.entities import GroundedAnswer
from backend.app.verification.citations import CitationParser
from backend.app.verification.claims import AtomicClaimExtractor
from backend.app.verification.models import (
    ClaimVerification,
    VerificationReport,
    VerifiedAnswer,
)
from backend.app.verification.support import SemanticSupportVerifier


class CitationVerificationService:
    """Verifies every generated claim and optionally repairs weak evidence through injected callbacks."""

    def __init__(
        self,
        support_verifier: SemanticSupportVerifier,
        min_coverage: float = 1.0,
        max_retries: int = 1,
    ) -> None:
        if not 0 <= min_coverage <= 1:
            raise ValueError("minimum coverage must be between 0 and 1.")
        if max_retries < 0:
            raise ValueError("verification max retries must not be negative.")
        self.support_verifier = support_verifier
        self.min_coverage = min_coverage
        self.max_retries = max_retries
        self.claim_extractor = AtomicClaimExtractor()
        self.citation_parser = CitationParser()

    def verify(
        self,
        answer: GroundedAnswer,
        retry_callback: Callable[[str], AgentState] | None = None,
        generation_callback: Callable[[AgentState], GroundedAnswer] | None = None,
    ) -> VerifiedAnswer:
        """Verify, optionally re-retrieve/re-generate targeted evidence, then filter unsupported claims."""
        current = answer
        retry_count = 0
        retry_triggered = False
        while True:
            report = self._verify_once(current, retry_triggered, retry_count)
            if current.status != "answer":
                return self._filtered(current, report)
            if not report.unsupported_claim_ids and not report.unsupported_citations:
                return self._verified(
                    current, report, "repaired" if retry_triggered else "verified"
                )
            if (
                retry_count >= self.max_retries
                or retry_callback is None
                or generation_callback is None
            ):
                return self._filtered(current, report)
            unsupported_text = self._unsupported_text(report, current)
            retry_state = retry_callback(unsupported_text)
            current = generation_callback(retry_state)
            retry_count += 1
            retry_triggered = True

    def _verify_once(
        self, answer: GroundedAnswer, retry_triggered: bool, retry_count: int
    ) -> VerificationReport:
        claims = self.claim_extractor.extract(answer.answer)
        evidence_by_marker = {
            marker: candidate for marker, candidate in enumerate(answer.evidence, start=1)
        }
        verifications: list[ClaimVerification] = []
        unsupported_citations: set[int] = set()
        for claim in claims:
            invalid = {
                marker for marker in claim.citation_markers if marker not in evidence_by_marker
            }
            unsupported_citations.update(invalid)
            cited_candidates = [
                evidence_by_marker[marker]
                for marker in claim.citation_markers
                if marker in evidence_by_marker
            ]
            scores = self.support_verifier.score(
                claim.text, [candidate.candidate.chunk.text for candidate in cited_candidates]
            )
            if invalid:
                supported = False
                reason = "citation references an evidence marker that does not exist"
            elif not claim.citation_markers:
                supported = False
                reason = "claim has no citation"
            elif not self.support_verifier.supports(scores):
                supported = False
                reason = "cited evidence does not meet the semantic support threshold"
            else:
                supported = True
                reason = "all cited evidence passages meet the semantic support threshold"
            verifications.append(
                ClaimVerification(
                    claim.claim_id,
                    claim.text,
                    claim.citation_markers,
                    tuple(scores),
                    supported,
                    reason,
                )
            )
        unsupported_claims = tuple(
            verification.claim_id for verification in verifications if not verification.supported
        )
        coverage = (
            sum(verification.supported for verification in verifications) / len(verifications)
            if verifications
            else 0.0
        )
        all_citations = self.citation_parser.parse(answer.answer)
        valid_citations = sum(marker in evidence_by_marker for marker in all_citations)
        citation_validity = valid_citations / len(all_citations) if all_citations else 0.0
        grounding = round((0.7 * coverage) + (0.3 * citation_validity), 4)
        return VerificationReport(
            claims=tuple(verifications),
            unsupported_claim_ids=unsupported_claims,
            unsupported_citations=tuple(sorted(unsupported_citations)),
            citation_coverage_score=round(coverage, 4),
            grounding_score=grounding,
            retry_triggered=retry_triggered,
            retry_count=retry_count,
        )

    def _verified(
        self, answer: GroundedAnswer, report: VerificationReport, status: str
    ) -> VerifiedAnswer:
        return VerifiedAnswer(
            answer=answer.answer,
            status=status,
            confidence_score=round(answer.confidence_score * report.grounding_score, 4),
            citation_coverage_score=report.citation_coverage_score,
            grounding_score=report.grounding_score,
            report=report,
            evidence=answer.evidence,
            rewritten_query=answer.rewritten_query,
            prompt_version=answer.prompt_version,
            model_name=answer.model_name,
            source_answer=answer,
        )

    def _filtered(self, answer: GroundedAnswer, report: VerificationReport) -> VerifiedAnswer:
        supported_claims = [claim for claim in report.claims if claim.supported]
        final_text = " ".join(
            claim.text + " " + " ".join(f"[{m}]" for m in claim.citation_markers)
            for claim in supported_claims
        ).strip()
        if not final_text or report.citation_coverage_score < self.min_coverage:
            final_text = "Insufficient evidence."
            status = "insufficient_evidence"
        else:
            status = "repaired"
        repaired = GroundedAnswer(
            answer=final_text,
            status="answer" if status == "repaired" else "insufficient_evidence",
            confidence_score=answer.confidence_score,
            evidence_coverage_score=answer.evidence_coverage_score,
            citations=answer.citations,
            evidence=answer.evidence,
            rewritten_query=answer.rewritten_query,
            prompt_version=answer.prompt_version,
            model_name=answer.model_name,
        )
        return self._verified(repaired, report, status)

    @staticmethod
    def _unsupported_text(report: VerificationReport, answer: GroundedAnswer) -> str:
        claims = {claim.claim_id: claim.text for claim in report.claims}
        return (
            " ".join(claims[claim_id] for claim_id in report.unsupported_claim_ids) or answer.answer
        )
