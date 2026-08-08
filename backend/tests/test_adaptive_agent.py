"""Unit and integration coverage for the bounded LangGraph retrieval agent."""

from datetime import UTC, datetime

from backend.app.agent.graph import AdaptiveRetrievalAgent
from backend.app.domain.entities import (
    ChunkRecord,
    HybridRetrievalCandidate,
    HybridRetrievalResult,
    RerankedCandidate,
    RerankingResult,
    RetrievalFilters,
)


def _hybrid_result(*, semantic: float = 0.9, keyword: float = 0.8, normalized: float = 0.9) -> HybridRetrievalResult:
    chunk = ChunkRecord(
        id="doc:p1:c1",
        document_id="doc",
        filename="source.pdf",
        page_number=1,
        sequence=1,
        text="Evidence passage",
        upload_timestamp=datetime.now(UTC),
    )
    candidate = HybridRetrievalCandidate(
        chunk=chunk,
        semantic_score=semantic,
        keyword_score=keyword,
        rrf_score=0.02,
        normalized_score=normalized,
    )
    return HybridRetrievalResult((candidate,), 0.5, 1.0, 1.0, 2.0)


class FakeRetrieval:
    def __init__(self, *results: HybridRetrievalResult) -> None:
        self.results = list(results) or [_hybrid_result()]
        self.queries: list[str] = []

    def retrieve(self, query: str, filters: RetrievalFilters, top_k: int) -> HybridRetrievalResult:
        self.queries.append(query)
        index = min(len(self.queries) - 1, len(self.results) - 1)
        return self.results[index]


class FakeReranking:
    def __init__(self, confidences: list[float]) -> None:
        self.confidences = confidences
        self.calls = 0

    def rerank(self, query: str, result: HybridRetrievalResult, top_k: int) -> RerankingResult:
        confidence = self.confidences[min(self.calls, len(self.confidences) - 1)]
        self.calls += 1
        candidate = result.candidates[0]
        reranked = RerankedCandidate(candidate, confidence * 10, confidence, 1)
        return RerankingResult((reranked,), confidence, 0.1, "test-reranker", 4)


class RepeatingRewriter:
    """Used to verify that identical rewrites cannot create an infinite graph loop."""

    def rewrite(self, query: str, attempt: int) -> str:
        return query


def test_agent_rewrites_then_returns_evidence_with_structured_trace() -> None:
    weak = _hybrid_result(semantic=0.2, keyword=0.15, normalized=0.18)
    strong = _hybrid_result()
    retrieval = FakeRetrieval(weak, strong)
    reranking = FakeReranking([0.1, 0.95])
    agent = AdaptiveRetrievalAgent(retrieval, reranking)

    state = agent.run(
        "cat behavior", RetrievalFilters(), top_k=1, confidence_threshold=0.7, max_retries=2
    )

    assert state["status"] == "evidence"
    assert state["rewritten_query"] is not None
    assert len(retrieval.queries) == 2
    assert state["attempt"] == 1
    assert [step["node"] for step in state["trace"]] == [
        "hybrid_retrieval",
        "cross_encoder_reranking",
        "evaluate_confidence",
        "rewrite_query",
        "retry_retrieval",
        "hybrid_retrieval",
        "cross_encoder_reranking",
        "evaluate_confidence",
    ]
    assert state["reasoning_steps"]


def test_agent_stops_at_retry_budget_with_insufficient_evidence() -> None:
    retrieval = FakeRetrieval()
    reranking = FakeReranking([0.0])
    agent = AdaptiveRetrievalAgent(retrieval, reranking)

    state = agent.run(
        "unknown", RetrievalFilters(), top_k=1, confidence_threshold=0.8, max_retries=2
    )

    assert state["status"] == "insufficient_evidence"
    assert state["stop_reason"] == "retry_budget_exhausted"
    assert state["attempt"] == 2
    assert len(retrieval.queries) == 3
    assert len(state["trace"]) == 13


def test_agent_prevents_repeated_rewrite_loops() -> None:
    retrieval = FakeRetrieval()
    agent = AdaptiveRetrievalAgent(
        retrieval, FakeReranking([0.0]), query_rewriter=RepeatingRewriter()
    )

    state = agent.run(
        "same query", RetrievalFilters(), top_k=1, confidence_threshold=0.8, max_retries=5
    )

    assert state["status"] == "insufficient_evidence"
    assert state["stop_reason"] == "query_rewrite_repeated"
    assert len(retrieval.queries) == 1
    assert state["trace"][-1]["node"] == "rewrite_query"
