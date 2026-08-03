"""Cross-encoder reranking over hybrid retrieval evidence only."""

import math
from time import perf_counter

from backend.app.domain.entities import HybridRetrievalResult, RerankedCandidate, RerankingResult
from backend.app.infrastructure.reranker.cross_encoder import CrossEncoderProvider


class RerankingService:
    """Reorders hybrid candidates with the original query and preserves every retrieval score."""

    def __init__(
        self,
        provider: CrossEncoderProvider,
        model_name: str,
        batch_size: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        self.provider = provider
        self.model_name = model_name
        self.batch_size = batch_size

    def rerank(
        self, query: str, retrieval_result: HybridRetrievalResult, top_k: int
    ) -> RerankingResult:
        """Score a hybrid result set in one batch and return the strongest evidence first."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        candidates = list(retrieval_result.candidates)
        if not candidates:
            return RerankingResult(
                candidates=(),
                confidence_score=0.0,
                latency_ms=0.0,
                model_name=self.model_name,
                batch_size=self.batch_size,
            )
        started_at = perf_counter()
        pairs = [(cleaned_query, candidate.chunk.text) for candidate in candidates]
        logits = self.provider.predict(pairs, batch_size=self.batch_size)
        if len(logits) != len(candidates):
            raise ValueError("Reranker returned a score count different from the candidate count.")
        ranked = sorted(
            zip(candidates, logits, strict=True), key=lambda pair: pair[1], reverse=True
        )[:top_k]
        normalized_scores = [_sigmoid(logit) for _, logit in ranked]
        reranked = tuple(
            RerankedCandidate(
                candidate=candidate,
                reranker_score=float(logit),
                normalized_reranker_score=normalized_score,
                rank=rank,
            )
            for rank, ((candidate, logit), normalized_score) in enumerate(
                zip(ranked, normalized_scores, strict=True), start=1
            )
        )
        return RerankingResult(
            candidates=reranked,
            confidence_score=_confidence(normalized_scores),
            latency_ms=round((perf_counter() - started_at) * 1000, 2),
            model_name=self.model_name,
            batch_size=self.batch_size,
        )


def _sigmoid(value: float) -> float:
    """Map an arbitrary cross-encoder logit into a stable confidence-like [0, 1] score."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _confidence(scores: list[float]) -> float:
    """Combine top score and separation from the runner-up as reranker confidence."""
    if not scores:
        return 0.0
    margin = scores[0] - scores[1] if len(scores) > 1 else scores[0]
    return round(min(1.0, max(0.0, (0.75 * scores[0]) + (0.25 * margin))), 4)
