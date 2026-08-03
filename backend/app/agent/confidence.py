"""Retrieval confidence evaluation independent of generation or citation verification."""

from backend.app.domain.entities import RerankingResult


class RetrievalConfidenceEvaluator:
    """Combines reranker confidence, source agreement, and evidence coverage."""

    def evaluate(self, result: RerankingResult) -> float:
        """Return a bounded confidence score suitable for the graph decision edge."""
        if not result.candidates:
            return 0.0
        top = result.candidates[0]
        agreement = float(top.candidate.semantic_score is not None) * 0.5
        agreement += float(top.candidate.keyword_score is not None) * 0.5
        coverage = min(1.0, len(result.candidates) / 3.0)
        score = (0.7 * result.confidence_score) + (0.2 * agreement) + (0.1 * coverage)
        return round(min(1.0, max(0.0, score)), 4)
