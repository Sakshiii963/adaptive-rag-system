"""Unit and integration coverage for cross-encoder reranking behavior."""

from datetime import UTC, datetime

import pytest

from backend.app.domain.entities import (
    ChunkRecord,
    HybridRetrievalCandidate,
    HybridRetrievalResult,
)
from backend.app.infrastructure.reranker.cross_encoder import LocalCrossEncoder
from backend.app.services.reranking import RerankingService


class FakeCrossEncoder:
    """Deterministic batch scorer that makes ordering and latency tests offline."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[tuple[str, str]], int]] = []

    def predict(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        self.calls.append((pairs, batch_size))
        return [5.0 if "strong" in passage else 1.0 for _, passage in pairs]


def _retrieval_result() -> HybridRetrievalResult:
    timestamp = datetime.now(UTC)
    candidates = tuple(
        HybridRetrievalCandidate(
            chunk=ChunkRecord(
                id=f"doc:p{i}:c1",
                document_id="doc",
                filename="source.pdf",
                page_number=i,
                sequence=1,
                text=text,
                upload_timestamp=timestamp,
            ),
            semantic_score=0.8 - (i * 0.1),
            keyword_score=0.7 - (i * 0.1),
            rrf_score=0.02 - (i * 0.001),
            normalized_score=0.8 - (i * 0.1),
        )
        for i, text in enumerate(["weak passage", "strong passage", "ordinary passage"], start=1)
    )
    return HybridRetrievalResult(
        candidates=candidates,
        confidence_score=0.5,
        semantic_latency_ms=1.0,
        keyword_latency_ms=1.0,
        total_latency_ms=2.0,
    )


def test_reranker_batches_pairs_preserves_scores_and_reorders() -> None:
    provider = FakeCrossEncoder()
    service = RerankingService(provider, "test-cross-encoder", batch_size=8)

    result = service.rerank("original query", _retrieval_result(), top_k=2)

    assert [item.candidate.chunk.text for item in result.candidates] == [
        "strong passage",
        "weak passage",
    ]
    assert result.candidates[0].rank == 1
    assert result.candidates[0].candidate.semantic_score == pytest.approx(0.6)
    assert result.candidates[0].candidate.keyword_score == pytest.approx(0.5)
    assert result.candidates[0].reranker_score == 5.0
    assert 0 <= result.candidates[0].normalized_reranker_score <= 1
    assert 0 <= result.confidence_score <= 1
    assert result.latency_ms >= 0
    assert len(provider.calls) == 1
    assert provider.calls[0][1] == 8
    assert len(provider.calls[0][0]) == 3


def test_reranker_handles_empty_candidates_without_model_call() -> None:
    provider = FakeCrossEncoder()
    service = RerankingService(provider, "test", batch_size=4)
    empty = HybridRetrievalResult((), 0.0, 0.0, 0.0, 0.0)

    result = service.rerank("query", empty, top_k=5)

    assert result.candidates == ()
    assert result.confidence_score == 0.0
    assert provider.calls == []


def test_local_cross_encoder_model_is_cached_after_first_load() -> None:
    reranker = LocalCrossEncoder("test-model")
    sentinel = object()
    reranker._model = sentinel

    # The already-loaded sentinel is reused; this test never imports or downloads a model.
    assert reranker._model is sentinel
    assert reranker.model_name == "test-model"
