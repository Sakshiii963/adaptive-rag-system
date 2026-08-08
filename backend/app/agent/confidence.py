"""Retrieval confidence evaluation independent of generation or citation verification."""

from dataclasses import dataclass

from backend.app.domain.entities import RerankingResult
from backend.app.services.retrieval import _confidence as hybrid_retrieval_confidence


@dataclass(frozen=True, slots=True)
class ConfidenceBreakdown:
    """Every component that contributes to the final retrieval confidence score."""

    semantic_score: float
    keyword_score: float
    retrieval_normalized_score: float
    retrieval_confidence: float
    top_reranker_score: float
    mean_reranker_score: float
    reranker_margin: float
    reranker_confidence: float
    agreement: float
    chunk_count: int
    document_count: int
    chunk_coverage: float
    document_coverage: float
    stage_mismatch: bool
    multi_document_synthesis: bool
    formula: str
    score: float


class RetrievalConfidenceEvaluator:
    """Fuses retrieval-stage and reranker-stage confidence for graph routing."""

    _STAGE_MISMATCH_RERANKER_CEILING = 0.35
    _STAGE_MISMATCH_RETRIEVAL_FLOOR = 0.5
    _SYNTHESIS_NORMALIZED_FLOOR = 0.5

    def evaluate(self, result: RerankingResult) -> float:
        """Return a bounded confidence score suitable for the graph decision edge."""
        return self.breakdown(result).score

    def breakdown(self, result: RerankingResult) -> ConfidenceBreakdown:
        """Return the full confidence equation and every contributing component."""
        if not result.candidates:
            return ConfidenceBreakdown(
                semantic_score=0.0,
                keyword_score=0.0,
                retrieval_normalized_score=0.0,
                retrieval_confidence=0.0,
                top_reranker_score=0.0,
                mean_reranker_score=0.0,
                reranker_margin=0.0,
                reranker_confidence=0.0,
                agreement=0.0,
                chunk_count=0,
                document_count=0,
                chunk_coverage=0.0,
                document_coverage=0.0,
                stage_mismatch=False,
                multi_document_synthesis=False,
                formula="none",
                score=0.0,
            )

        candidates = result.candidates
        hybrid_candidates = [item.candidate for item in candidates]
        top = hybrid_candidates[0]
        reranker_scores = [item.normalized_reranker_score for item in candidates]
        top_reranker = reranker_scores[0]
        mean_reranker = sum(reranker_scores) / len(reranker_scores)
        reranker_margin = (
            reranker_scores[0] - reranker_scores[1] if len(reranker_scores) > 1 else top_reranker
        )

        semantic = top.semantic_score or 0.0
        keyword = top.keyword_score or 0.0
        retrieval_normalized = top.normalized_score
        retrieval_conf = hybrid_retrieval_confidence(hybrid_candidates)
        reranker_conf = result.confidence_score

        agreement = (
            float(top.semantic_score is not None) * 0.5
            + float(top.keyword_score is not None) * 0.5
        )

        chunk_count = len(candidates)
        document_count = len({item.candidate.chunk.document_id for item in candidates})
        chunk_coverage = min(1.0, chunk_count / 3.0)
        document_coverage = min(1.0, document_count / 2.0)

        multi_document_synthesis = document_count >= 2 and all(
            candidate.normalized_score >= self._SYNTHESIS_NORMALIZED_FLOOR
            for candidate in hybrid_candidates
        )
        stage_mismatch = (
            retrieval_conf >= self._STAGE_MISMATCH_RETRIEVAL_FLOOR
            and reranker_conf < self._STAGE_MISMATCH_RERANKER_CEILING
            and (document_count == 1 or multi_document_synthesis)
        )

        if stage_mismatch or multi_document_synthesis:
            # Retrieval found evidence but the cross-encoder under-scored query-passage pairs.
            # Reuse the hybrid retrieval confidence formula weights (0.55/0.30/0.15) as the
            # dominant signal and keep reranker as a secondary check.
            formula = (
                "0.55*retrieval_confidence + 0.15*reranker_confidence + 0.15*agreement"
                " + 0.075*chunk_coverage + 0.075*document_coverage"
            )
            score = (
                (0.55 * retrieval_conf)
                + (0.15 * reranker_conf)
                + (0.15 * agreement)
                + (0.075 * chunk_coverage)
                + (0.075 * document_coverage)
            )
        else:
            # Single-best-match gating: unchanged from the original evaluator.
            formula = "0.70*reranker_confidence + 0.20*agreement + 0.10*chunk_coverage"
            score = (0.70 * reranker_conf) + (0.20 * agreement) + (0.10 * chunk_coverage)
            if retrieval_conf < self._STAGE_MISMATCH_RETRIEVAL_FLOOR:
                # Reranker margin alone cannot override a weak first-stage retrieval signal.
                score = min(score, retrieval_conf + 0.1)
                formula += "; capped by retrieval_confidence + 0.1 when retrieval_confidence < 0.5"

        return ConfidenceBreakdown(
            semantic_score=round(semantic, 4),
            keyword_score=round(keyword, 4),
            retrieval_normalized_score=round(retrieval_normalized, 4),
            retrieval_confidence=round(retrieval_conf, 4),
            top_reranker_score=round(top_reranker, 4),
            mean_reranker_score=round(mean_reranker, 4),
            reranker_margin=round(reranker_margin, 4),
            reranker_confidence=round(reranker_conf, 4),
            agreement=round(agreement, 4),
            chunk_count=chunk_count,
            document_count=document_count,
            chunk_coverage=round(chunk_coverage, 4),
            document_coverage=round(document_coverage, 4),
            stage_mismatch=stage_mismatch,
            multi_document_synthesis=multi_document_synthesis,
            formula=formula,
            score=round(min(1.0, max(0.0, score)), 4),
        )


def _confidence_components(breakdown: ConfidenceBreakdown) -> dict[str, object]:
    """Render breakdown fields for structured logging."""
    return {
        "semantic_score": breakdown.semantic_score,
        "keyword_score": breakdown.keyword_score,
        "retrieval_normalized_score": breakdown.retrieval_normalized_score,
        "retrieval_confidence": breakdown.retrieval_confidence,
        "top_reranker_score": breakdown.top_reranker_score,
        "mean_reranker_score": breakdown.mean_reranker_score,
        "reranker_margin": breakdown.reranker_margin,
        "reranker_confidence": breakdown.reranker_confidence,
        "agreement": breakdown.agreement,
        "chunk_count": breakdown.chunk_count,
        "document_count": breakdown.document_count,
        "chunk_coverage": breakdown.chunk_coverage,
        "document_coverage": breakdown.document_coverage,
        "stage_mismatch": breakdown.stage_mismatch,
        "multi_document_synthesis": breakdown.multi_document_synthesis,
        "confidence_equation": breakdown.formula,
    }
