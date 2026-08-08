"""Unit and integration coverage for claim, citation, support, and repair verification."""

from dataclasses import replace
from datetime import UTC, datetime

from backend.app.domain.entities import (
    ChunkRecord,
    GroundedAnswer,
    GroundedCitation,
    HybridRetrievalCandidate,
    RerankedCandidate,
)
from backend.app.verification.claims import AtomicClaimExtractor
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


class DocumentAwareSupportScorer:
    """Small deterministic scorer that requires claim terms in the cited passage."""

    def predict(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        return [
            5.0
            if any(term in claim.lower() and term in passage.lower() for term in ("adaptive", "audit"))
            else -5.0
            for claim, passage in pairs
        ]


def test_claim_extractor_preserves_abbreviations_decimals_quotes_and_parentheses() -> None:
    claims = AtomicClaimExtractor().extract(
        'APFD stands for Area Under the Percent Faults vs. Detected Curve (v2.5) [1]. '
        'The reviewer said "valid" [2].'
    )

    assert len(claims) == 2
    assert "vs. Detected Curve (v2.5)" in claims[0].text
    assert claims[0].citation_markers == (1,)
    assert claims[1].text == 'The reviewer said "valid".'
    assert claims[1].citation_markers == (2,)


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


def _two_document_answer(text: str) -> GroundedAnswer:
    timestamp = datetime.now(UTC)
    items = []
    for index, (document_id, filename, passage) in enumerate(
        (
            ("doc-a", "adaptive-retrieval.pdf", "Adaptive retrieval retries weak queries."),
            ("doc-b", "verification-policy.pdf", "Audit records are retained for seven years."),
        ),
        start=1,
    ):
        items.append(
            RerankedCandidate(
                HybridRetrievalCandidate(
                    ChunkRecord(
                        f"{document_id}:p1:c1",
                        document_id,
                        filename,
                        1,
                        1,
                        passage,
                        timestamp,
                    ),
                    0.9,
                    0.8,
                    0.02,
                    0.9,
                ),
                4.0,
                0.98,
                index,
            )
        )
    evidence = tuple(items)
    return GroundedAnswer(
        answer=text,
        status="answer",
        confidence_score=0.9,
        evidence_coverage_score=1.0,
        citations=tuple(GroundedCitation(index, item) for index, item in enumerate(evidence, 1)),
        evidence=evidence,
        rewritten_query=None,
        prompt_version="v1",
        model_name="qwen2.5:7b",
    )


def test_supported_claim_references_existing_chunk() -> None:
    result = _service(FakeSupportScorer()).verify(_answer("Cats are supported [1]."))

    assert result.status == "verified"
    assert result.report.unsupported_claim_ids == ()
    assert result.report.unsupported_citations == ()
    assert result.citation_coverage_score == 1.0
    assert result.grounding_score == 1.0


def test_grouped_multiple_citations_are_mapped_and_verified() -> None:
    result = _service(FakeSupportScorer()).verify(_answer("Cats are supported [1, 1]."))

    assert result.status == "verified"
    assert result.report.claims[0].citation_markers == (1, 1)


def test_supported_claim_with_citation_on_following_line_is_preserved() -> None:
    result = _service(FakeSupportScorer()).verify(_answer("Cats are supported.\n[1]"))

    assert result.status == "verified"
    assert result.report.unsupported_claim_ids == ()
    assert result.report.claims[0].citation_markers == (1,)


def test_citation_stays_with_claim_when_next_claim_follows() -> None:
    result = _service(FakeSupportScorer()).verify(
        _answer("Cats are supported. [1] Dogs are unrelated. [1]")
    )

    assert result.status == "insufficient_evidence"
    assert [claim.citation_markers for claim in result.report.claims] == [(1,), (1,)]
    assert result.report.unsupported_claim_ids == (2,)


def test_trailing_citations_support_multi_sentence_summary() -> None:
    result = _service(FakeSupportScorer()).verify(
        _answer("Cats are supported. Cats are supported. [1]")
    )

    assert result.status == "verified"
    assert [claim.citation_markers for claim in result.report.claims] == [(1,), (1,)]


def test_trailing_summary_citation_does_not_hide_hallucinated_sentence() -> None:
    result = _service(FakeSupportScorer()).verify(
        _answer("Cats are supported. Dogs are unrelated. [1]")
    )

    assert result.status == "insufficient_evidence"
    assert result.report.unsupported_claim_ids == (2,)


def test_uncited_claim_is_rejected() -> None:
    result = _service(FakeSupportScorer()).verify(_answer("Cats are supported."))

    assert result.status == "insufficient_evidence"
    assert result.answer == "Insufficient evidence."
    assert result.report.unsupported_claim_ids == (1,)
    assert result.report.claims[0].reason == "claim has no citation"


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


def test_two_document_synthesis_uses_explicit_marker_candidates() -> None:
    result = _service(DocumentAwareSupportScorer()).verify(
        _two_document_answer(
            "Adaptive retrieval retries weak queries [1]. Audit records are retained for seven years [2]."
        )
    )

    assert result.status == "verified"
    assert result.report.unsupported_claim_ids == ()
    assert result.evidence[0].candidate.chunk.document_id == "doc-a"
    assert result.evidence[1].candidate.chunk.document_id == "doc-b"


def test_two_document_paragraph_summary_assigns_trailing_markers_by_sentence_order() -> None:
    result = _service(DocumentAwareSupportScorer()).verify(
        _two_document_answer(
            "Adaptive retrieval retries weak queries. Audit records are retained for seven years. [1] [2]"
        )
    )

    assert result.status == "verified"
    assert [claim.citation_markers for claim in result.report.claims] == [(1,), (2,)]


def test_mixed_document_synthesis_rejects_wrong_marker_support() -> None:
    result = _service(DocumentAwareSupportScorer()).verify(
        _two_document_answer("Audit records are retained for seven years [1].")
    )

    assert result.status == "insufficient_evidence"
    assert result.report.unsupported_claim_ids == (1,)


def test_marker_map_mismatch_is_rejected_before_support_can_pass() -> None:
    answer = _two_document_answer("Adaptive retrieval retries weak queries [1].")
    wrong_mapping = replace(
        answer,
        citations=(
            GroundedCitation(1, answer.citations[1].candidate),
            GroundedCitation(2, answer.citations[0].candidate),
        ),
    )

    result = _service(DocumentAwareSupportScorer()).verify(wrong_mapping)

    assert result.status == "insufficient_evidence"
    assert result.report.unsupported_citations == (1,)
