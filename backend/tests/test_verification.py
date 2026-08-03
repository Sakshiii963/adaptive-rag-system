"""Unit and integration coverage for claim, citation, support, and repair verification."""

from datetime import UTC, datetime

from backend.app.domain.entities import (
    ChunkRecord,
    GroundedAnswer,
    HybridRetrievalCandidate,
    RerankedCandidate,
)
from backend.app.verification.service import CitationVerificationService
from backend.app.verification.support import SemanticSupportVerifier


class FakeSupportScorer:
    """Deterministic semantic scorer used without downloading or invoking a model."""

    def __init__(self, supported_terms: tuple[str, ...] = ("cats", "supported")) -> None:
        self.supported_terms = supported_terms
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        self.calls.append(pairs)
        return [
            5.0 if any(term in claim.lower() for term in self.supported_terms) else -5.0
            for claim, _ in pairs
        ]


def _answer(text: str, evidence_count: int = 1) -> GroundedAnswer:
    timestamp = datetime.now(UTC)
    evidence = tuple(
        RerankedCandidate(
            HybridRetrievalCandidate(
                ChunkRecord(
                    f"doc:p{i}:c1",
                    "doc",
                    "source.pdf",
                    i,
                    1,
                    "Cats are supported evidence." if i == 1 else "Other evidence.",
                    timestamp,
                ),
                0.9,
                0.8,
                0.02,
                0.9,
            ),
            4.0,
            0.98,
            i,
        )
        for i in range(1, evidence_count + 1)
    )
    return GroundedAnswer(
        answer=text,
        status="answer",
        confidence_score=0.9,
        evidence_coverage_score=1.0,
        citations=(),
        evidence=evidence,
        rewritten_query=None,
        prompt_version="v1",
        model_name="qwen2.5:7b",
    )


def _service(scorer: FakeSupportScorer, max_retries: int = 0) -> CitationVerificationService:
    return CitationVerificationService(
        SemanticSupportVerifier(scorer, threshold=0.65, batch_size=8),
        min_coverage=1.0,
        max_retries=max_retries,
    )


def test_supported_claim_references_existing_chunk() -> None:
    result = _service(FakeSupportScorer()).verify(_answer("Cats are supported [1]."))

    assert result.status == "verified"
    assert result.report.unsupported_claim_ids == ()
    assert result.report.unsupported_citations == ()
    assert result.citation_coverage_score == 1.0
    assert result.grounding_score == 1.0


def test_unknown_citation_is_rejected() -> None:
    result = _service(FakeSupportScorer()).verify(_answer("Cats are supported [2]."))

    assert result.status == "insufficient_evidence"
    assert result.answer == "Insufficient evidence."
    assert result.report.unsupported_citations == (2,)
    assert result.report.claims[0].supported is False


def test_unsupported_claim_is_rejected_and_not_returned() -> None:
    result = _service(FakeSupportScorer()).verify(
        _answer("Cats are supported [1]. Dogs are unrelated [1].")
    )

    assert result.status == "insufficient_evidence"
    assert result.report.unsupported_claim_ids == (2,)
    assert "Dogs are unrelated" not in result.answer


def test_unsupported_claim_triggers_targeted_retrieval_and_repair() -> None:
    scorer = FakeSupportScorer(supported_terms=("repaired",))
    service = _service(scorer, max_retries=1)
    retry_queries: list[str] = []

    def retry(query: str) -> dict:
        retry_queries.append(query)
        return {"status": "evidence"}

    def regenerate(state: dict) -> GroundedAnswer:
        return _answer("Repaired evidence is supported [1].")

    result = service.verify(_answer("Unsupported fact [1]."), retry, regenerate)

    assert result.status == "repaired"
    assert result.answer == "Repaired evidence is supported [1]."
    assert result.report.retry_triggered is True
    assert result.report.retry_count == 1
    assert retry_queries == ["Unsupported fact."]
