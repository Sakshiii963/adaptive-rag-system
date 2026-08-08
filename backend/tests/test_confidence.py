"""Regression and audit coverage for retrieval confidence evaluation."""

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from backend.app.agent.confidence import RetrievalConfidenceEvaluator
from backend.app.agent.graph import AdaptiveRetrievalAgent
from backend.app.domain.entities import (
    ChunkRecord,
    GroundedAnswer,
    HybridRetrievalCandidate,
    HybridRetrievalResult,
    RerankedCandidate,
    RerankingResult,
    RetrievalFilters,
)
from backend.app.generation.context import ContextWindowManager
from backend.app.generation.prompt import GroundedPromptBuilder
from backend.app.generation.service import GroundedGenerationService
from backend.app.services.reranking import _confidence as reranker_aggregate_confidence

THRESHOLD = 0.65

DOC_A_TEXT = (
    "Adaptive retrieval evaluates the quality of retrieved evidence before an answer is generated."
)
DOC_B_TEXT = "Audit records are retained for seven years from the date of record creation."
DOC_A = ("doc-a", "adaptive-retrieval-brief.pdf", DOC_A_TEXT)
DOC_B = ("doc-b", "verification-policy.pdf", DOC_B_TEXT)


def _old_confidence(result: RerankingResult) -> float:
    """Original pre-fix evaluator preserved for audit comparisons."""
    if not result.candidates:
        return 0.0
    top = result.candidates[0]
    agreement = float(top.candidate.semantic_score is not None) * 0.5
    agreement += float(top.candidate.keyword_score is not None) * 0.5
    coverage = min(1.0, len(result.candidates) / 3.0)
    score = (0.7 * result.confidence_score) + (0.2 * agreement) + (0.1 * coverage)
    return round(min(1.0, max(0.0, score)), 4)


def _chunk(document_id: str, filename: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        id=f"{document_id}:p1:c1",
        document_id=document_id,
        filename=filename,
        page_number=1,
        sequence=1,
        text=text,
        upload_timestamp=datetime.now(UTC),
    )


def _candidate(
    document_id: str,
    filename: str,
    text: str,
    *,
    semantic: float = 0.9,
    keyword: float = 0.8,
    normalized: float = 0.9,
) -> HybridRetrievalCandidate:
    return HybridRetrievalCandidate(
        chunk=_chunk(document_id, filename, text),
        semantic_score=semantic,
        keyword_score=keyword,
        rrf_score=0.02,
        normalized_score=normalized,
    )


def _reranking_result(
    rows: list[tuple[HybridRetrievalCandidate, float]],
) -> RerankingResult:
    reranker_scores = [score for _, score in rows]
    reranked = tuple(
        RerankedCandidate(candidate, score * 10, score, rank)
        for rank, (candidate, score) in enumerate(rows, start=1)
    )
    return RerankingResult(
        reranked,
        reranker_aggregate_confidence(reranker_scores),
        1.0,
        "test-reranker",
        8,
    )


@dataclass(frozen=True, slots=True)
class AuditScenario:
    query: str
    rows: list[tuple[HybridRetrievalCandidate, float]]
    expected_accepted: bool
    answer: str | None = None


def _scenario_rows(
    docs: list[tuple[str, str, str]],
    reranker_scores: list[float],
    *,
    semantic: float = 0.9,
    keyword: float = 0.8,
    normalized: float = 0.9,
) -> list[tuple[HybridRetrievalCandidate, float]]:
    return [
        (_candidate(doc_id, filename, text, semantic=semantic, keyword=keyword, normalized=normalized), score)
        for (doc_id, filename, text), score in zip(docs, reranker_scores, strict=True)
    ]


AUDIT_SCENARIOS = [
    AuditScenario(
        "What is adaptive retrieval?",
        _scenario_rows([DOC_A], [0.88]),
        True,
        "Adaptive retrieval evaluates retrieved evidence before generation. [1]",
    ),
    AuditScenario(
        "Explain adaptive retrieval.",
        _scenario_rows([DOC_A], [0.82]),
        True,
        "Adaptive retrieval evaluates retrieved evidence before generation. [1]",
    ),
    AuditScenario(
        "Summarize adaptive-retrieval-brief.pdf.",
        _scenario_rows([DOC_A], [0.14]),
        True,
        "Adaptive retrieval evaluates retrieved evidence before generation. [1]",
    ),
    AuditScenario(
        "Summarize both uploaded documents.",
        _scenario_rows([DOC_A, DOC_B], [0.12, 0.11]),
        True,
        (
            "Adaptive retrieval evaluates retrieved evidence before generation. [1] "
            "Audit records are retained for seven years. [2]"
        ),
    ),
    AuditScenario(
        "Compare the two uploaded documents.",
        _scenario_rows([DOC_A, DOC_B], [0.13, 0.12]),
        True,
        (
            "Adaptive retrieval evaluates evidence before generation. [1] "
            "Audit records are retained for seven years. [2]"
        ),
    ),
    AuditScenario(
        "What is the capital of France?",
        _scenario_rows(
            [DOC_A],
            [0.05],
            semantic=0.08,
            keyword=0.04,
            normalized=0.07,
        ),
        False,
        None,
    ),
]


@pytest.mark.parametrize("scenario", AUDIT_SCENARIOS, ids=lambda item: item.query)
def test_audit_old_vs_new_confidence(scenario: AuditScenario) -> None:
    result = _reranking_result(scenario.rows)
    evaluator = RetrievalConfidenceEvaluator()
    breakdown = evaluator.breakdown(result)
    old_score = _old_confidence(result)
    new_score = breakdown.score

    assert (new_score >= THRESHOLD) is scenario.expected_accepted
    if scenario.query == "Summarize both uploaded documents.":
        assert old_score < THRESHOLD
        assert breakdown.multi_document_synthesis is True
    if scenario.query == "What is the capital of France?":
        assert old_score < THRESHOLD
        assert new_score < THRESHOLD


def test_single_document_lookup_still_passes() -> None:
    result = _reranking_result([(_candidate(*DOC_A), 0.91)])
    assert RetrievalConfidenceEvaluator().evaluate(result) >= THRESHOLD


def test_single_document_summary_uses_stage_mismatch_path() -> None:
    result = _reranking_result([(_candidate(*DOC_A), 0.14)])
    breakdown = RetrievalConfidenceEvaluator().breakdown(result)

    assert breakdown.stage_mismatch is True
    assert breakdown.score >= THRESHOLD


def test_multi_document_summary_passes_despite_low_reranker_margin() -> None:
    result = _reranking_result(_scenario_rows([DOC_A, DOC_B], [0.12, 0.11]))
    breakdown = RetrievalConfidenceEvaluator().breakdown(result)

    assert breakdown.document_count == 2
    assert breakdown.reranker_confidence < THRESHOLD
    assert breakdown.multi_document_synthesis is True
    assert breakdown.score >= THRESHOLD


def test_comparison_query_passes_with_diverse_documents() -> None:
    result = _reranking_result(_scenario_rows([DOC_A, DOC_B], [0.15, 0.14]))
    assert RetrievalConfidenceEvaluator().evaluate(result) >= THRESHOLD


def test_synthesis_query_passes_when_retrieval_is_strong_and_reranker_is_flat() -> None:
    result = _reranking_result(
        _scenario_rows(
            [
                ("doc-a", "architecture.pdf", "The agent rewrites weak queries."),
                ("doc-b", "operations.pdf", "Audit logs capture retrieval traces."),
                ("doc-c", "security.pdf", "Verification rejects unsupported claims."),
            ],
            [0.09, 0.08, 0.08],
        )
    )
    breakdown = RetrievalConfidenceEvaluator().breakdown(result)

    assert breakdown.document_count == 3
    assert breakdown.mean_reranker_score < 0.15
    assert breakdown.score >= THRESHOLD


def test_unsupported_query_still_fails_with_weak_retrieval() -> None:
    result = _reranking_result(
        [
            (
                _candidate(
                    "doc-a",
                    "notes.pdf",
                    "Unrelated content.",
                    semantic=0.12,
                    keyword=0.05,
                    normalized=0.1,
                ),
                0.04,
            )
        ]
    )
    assert RetrievalConfidenceEvaluator().evaluate(result) < THRESHOLD


def test_mixed_relevant_and_irrelevant_documents_are_rejected() -> None:
    result = _reranking_result(
        [
            (_candidate(*DOC_A), 0.12),
            (
                _candidate(
                    "doc-noise",
                    "noise.pdf",
                    "Completely unrelated filler content.",
                    semantic=0.15,
                    keyword=0.05,
                    normalized=0.12,
                ),
                0.11,
            ),
        ]
    )
    breakdown = RetrievalConfidenceEvaluator().breakdown(result)

    assert breakdown.multi_document_synthesis is False
    assert breakdown.score < THRESHOLD


def test_hallucinated_query_with_strong_but_irrelevant_reranker_still_rejected() -> None:
    """A high reranker margin on irrelevant evidence must not bypass weak retrieval checks."""
    result = _reranking_result(
        [
            (
                _candidate(
                    "doc-a",
                    "noise.pdf",
                    "Mars is discussed in science fiction.",
                    semantic=0.1,
                    keyword=0.05,
                    normalized=0.08,
                ),
                0.82,
            ),
            (
                _candidate(
                    "doc-b",
                    "noise.pdf",
                    "Other unrelated filler.",
                    semantic=0.05,
                    keyword=0.02,
                    normalized=0.04,
                ),
                0.05,
            ),
        ]
    )
    assert RetrievalConfidenceEvaluator().evaluate(result) < THRESHOLD


class _StaticRetrieval:
    def __init__(self, result: HybridRetrievalResult) -> None:
        self.result = result

    def retrieve(self, query: str, filters: RetrievalFilters, top_k: int) -> HybridRetrievalResult:
        return self.result


class _StaticReranking:
    def __init__(self, result: RerankingResult) -> None:
        self.result = result

    def rerank(self, query: str, result: HybridRetrievalResult, top_k: int) -> RerankingResult:
        return self.result


class _FakeGenerationProvider:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def generate(self, prompt: str, max_output_tokens: int) -> str:
        return self.answer


def _hybrid_from_reranking(reranking: RerankingResult) -> HybridRetrievalResult:
    candidates = tuple(item.candidate for item in reranking.candidates)
    return HybridRetrievalResult(candidates, 0.9, 1.0, 1.0, 2.0)


def _run_agent(query: str, reranking: RerankingResult) -> dict:
    agent = AdaptiveRetrievalAgent(
        _StaticRetrieval(_hybrid_from_reranking(reranking)),
        _StaticReranking(reranking),
    )
    return agent.run(query, RetrievalFilters(), top_k=len(reranking.candidates), confidence_threshold=THRESHOLD, max_retries=0)


@pytest.mark.parametrize("scenario", AUDIT_SCENARIOS, ids=lambda item: item.query)
def test_end_to_end_agent_routing_matches_expectation(scenario: AuditScenario) -> None:
    reranking = _reranking_result(scenario.rows)
    state = _run_agent(scenario.query, reranking)
    accepted = state["status"] == "evidence"

    assert accepted is scenario.expected_accepted
    if scenario.expected_accepted:
        service = GroundedGenerationService(
            _FakeGenerationProvider(scenario.answer or "Grounded answer [1]."),
            "qwen2.5:7b",
            GroundedPromptBuilder(),
            ContextWindowManager(2000),
            128,
        )
        answer: GroundedAnswer = service.generate(state)
        assert answer.status == "answer"
        assert answer.answer != "Insufficient evidence."
    else:
        service = GroundedGenerationService(
            _FakeGenerationProvider("This must never be returned."),
            "qwen2.5:7b",
            GroundedPromptBuilder(),
            ContextWindowManager(2000),
            128,
        )
        answer = service.generate(state)
        assert answer.status == "insufficient_evidence"
        assert answer.answer == "Insufficient evidence."
