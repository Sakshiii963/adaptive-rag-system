"""Deterministic query-rewrite port for retrieval recovery without an LLM."""

from typing import Protocol


class QueryRewriter(Protocol):
    """Port for deterministic local query rewriting without an LLM."""

    def rewrite(self, query: str, attempt: int) -> str:
        """Return a semantically focused alternative query."""


class HeuristicQueryRewriter:
    """Adds bounded evidence-oriented terms, never calling Ollama or another model."""

    _suffixes = (
        "relevant evidence key concepts",
        "exact terminology definitions and supporting details",
        "specific facts dates entities and evidence",
    )

    def rewrite(self, query: str, attempt: int) -> str:
        """Create a deterministic, bounded rewrite for a numbered retry."""
        suffix = self._suffixes[min(max(attempt - 1, 0), len(self._suffixes) - 1)]
        return f"{query.strip()} {suffix}".strip()
