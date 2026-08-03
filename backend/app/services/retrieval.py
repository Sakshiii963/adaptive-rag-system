"""Parallel hybrid retrieval and deterministic result fusion; no agent or LLM behavior."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter

from backend.app.domain.entities import (
    ChunkRecord,
    HybridRetrievalCandidate,
    HybridRetrievalResult,
    RetrievalCandidate,
    RetrievalFilters,
)
from backend.app.infrastructure.bm25.index import BM25IndexManager
from backend.app.infrastructure.embedding.bge_embeddings import EmbeddingProvider
from backend.app.infrastructure.vector.chroma_store import ChromaChunkStore


class HybridRetrievalEngine:
    """Runs semantic and lexical retrieval concurrently, then fuses evidence with RRF."""

    def __init__(
        self,
        vector_store: ChromaChunkStore,
        embedding_provider: EmbeddingProvider,
        bm25_manager: BM25IndexManager,
        rrf_constant: int,
        candidate_multiplier: int,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.bm25_manager = bm25_manager
        self.rrf_constant = rrf_constant
        self.candidate_multiplier = candidate_multiplier

    def retrieve(self, query: str, filters: RetrievalFilters, top_k: int) -> HybridRetrievalResult:
        """Retrieve, deduplicate, normalize, fuse, and score evidence without generating an answer."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        candidate_limit = top_k * self.candidate_multiplier
        started_at = perf_counter()
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid-retrieval") as executor:
            semantic_future = executor.submit(
                self._semantic_search, cleaned_query, filters, candidate_limit
            )
            keyword_future = executor.submit(
                self._keyword_search, cleaned_query, filters, candidate_limit
            )
            semantic, semantic_latency_ms = semantic_future.result()
            keyword, keyword_latency_ms = keyword_future.result()
        candidates = _fuse_rrf(semantic, keyword, self.rrf_constant, top_k)
        total_latency_ms = round((perf_counter() - started_at) * 1000, 2)
        return HybridRetrievalResult(
            candidates=tuple(candidates),
            confidence_score=_confidence(candidates),
            semantic_latency_ms=semantic_latency_ms,
            keyword_latency_ms=keyword_latency_ms,
            total_latency_ms=total_latency_ms,
        )

    def _semantic_search(
        self, query: str, filters: RetrievalFilters, limit: int
    ) -> tuple[list[RetrievalCandidate], float]:
        started_at = perf_counter()
        embedding = self.embedding_provider.embed_query(query)
        candidates = self.vector_store.search(embedding, filters, limit)
        return candidates, round((perf_counter() - started_at) * 1000, 2)

    def _keyword_search(
        self, query: str, filters: RetrievalFilters, limit: int
    ) -> tuple[list[RetrievalCandidate], float]:
        started_at = perf_counter()
        candidates = self.bm25_manager.search(query, filters, limit)
        return candidates, round((perf_counter() - started_at) * 1000, 2)


def _fuse_rrf(
    semantic: list[RetrievalCandidate],
    keyword: list[RetrievalCandidate],
    rrf_constant: int,
    top_k: int,
) -> list[HybridRetrievalCandidate]:
    """Merge source rankings by chunk ID, retaining source scores for traceability."""
    merged: dict[str, _FusionRecord] = {}
    for source_candidates in (semantic, keyword):
        for rank, candidate in enumerate(source_candidates, start=1):
            record = merged.setdefault(candidate.chunk.id, _FusionRecord(chunk=candidate.chunk))
            if candidate.source == "semantic":
                record.semantic = candidate
            else:
                record.keyword = candidate
            record.rrf += 1.0 / (rrf_constant + rank)
    maximum_rrf = 2.0 / (rrf_constant + 1)
    fused = [
        HybridRetrievalCandidate(
            chunk=record.chunk,
            semantic_score=record.semantic.normalized_score if record.semantic else None,
            keyword_score=record.keyword.normalized_score if record.keyword else None,
            rrf_score=record.rrf,
            normalized_score=min(1.0, record.rrf / maximum_rrf),
        )
        for record in merged.values()
    ]
    fused.sort(key=lambda candidate: candidate.rrf_score, reverse=True)
    return fused[:top_k]


@dataclass
class _FusionRecord:
    """Mutable internal accumulator for one de-duplicated chunk ID."""

    chunk: ChunkRecord
    semantic: RetrievalCandidate | None = None
    keyword: RetrievalCandidate | None = None
    rrf: float = 0.0


def _confidence(candidates: list[HybridRetrievalCandidate]) -> float:
    """Estimate evidence quality from top fused strength, source agreement, and score separation."""
    if not candidates:
        return 0.0
    top = candidates[0]
    agreement = (
        float(top.semantic_score is not None) * 0.5 + float(top.keyword_score is not None) * 0.5
    )
    runner_up = candidates[1].normalized_score if len(candidates) > 1 else 0.0
    margin = max(0.0, top.normalized_score - runner_up)
    return round(min(1.0, (0.55 * top.normalized_score) + (0.3 * agreement) + (0.15 * margin)), 4)
