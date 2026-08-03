"""Semantic claim-to-evidence support verification."""

import math
from typing import Protocol


class SupportScorer(Protocol):
    """Port implemented by the existing local cross-encoder adapter."""

    def predict(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        """Return one relevance logit for each claim/evidence pair."""


class SemanticSupportVerifier:
    """Scores claim/evidence entailment-like relevance and normalizes logits to [0, 1]."""

    def __init__(self, scorer: SupportScorer, threshold: float, batch_size: int) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("support threshold must be between 0 and 1.")
        if batch_size < 1:
            raise ValueError("support batch size must be at least 1.")
        self.scorer = scorer
        self.threshold = threshold
        self.batch_size = batch_size

    def score(self, claim: str, evidence: list[str]) -> list[float]:
        """Batch-score one claim against its cited passages."""
        if not evidence:
            return []
        logits = self.scorer.predict([(claim, passage) for passage in evidence], self.batch_size)
        return [_sigmoid(value) for value in logits]

    def supports(self, scores: list[float]) -> bool:
        """Require every cited passage to meet the semantic support threshold."""
        return bool(scores) and min(scores) >= self.threshold


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)
