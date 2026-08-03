"""Unit and integration coverage for parallel semantic/BM25 hybrid retrieval."""

from datetime import UTC, datetime
from pathlib import Path

from backend.app.domain.entities import ChunkRecord, RetrievalFilters
from backend.app.infrastructure.bm25.index import BM25IndexManager
from backend.app.infrastructure.vector.chroma_store import ChromaChunkStore
from backend.app.services.retrieval import HybridRetrievalEngine


class RetrievalEmbeddingProvider:
    """Small deterministic vector space for fast, local retrieval assertions."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        lowered = query.lower()
        if "cat" in lowered or "feline" in lowered:
            return [1.0, 0.0, 0.0]
        if "database" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def test_hybrid_retrieval_fuses_sources_filters_metadata_and_measures_latency(
    tmp_path: Path,
) -> None:
    provider = RetrievalEmbeddingProvider()
    vector_store = ChromaChunkStore(str(tmp_path / "chroma"), "hybrid_chunks")
    bm25 = BM25IndexManager()
    timestamp = datetime.now(UTC)
    cat_chunk = ChunkRecord(
        id="doc-cats:p1:c1",
        document_id="doc-cats",
        filename="cats.pdf",
        page_number=1,
        sequence=1,
        text="Cats are domestic feline animals with distinct behavior.",
        upload_timestamp=timestamp,
    )
    database_chunk = ChunkRecord(
        id="doc-data:p2:c1",
        document_id="doc-data",
        filename="database.pdf",
        page_number=2,
        sequence=1,
        text="A database persists structured application records.",
        upload_timestamp=timestamp,
    )
    vector_store.replace_document([cat_chunk], provider.embed_documents([cat_chunk.text]))
    vector_store.replace_document([database_chunk], provider.embed_documents([database_chunk.text]))
    bm25.rebuild_document_index(cat_chunk.document_id, [cat_chunk])
    bm25.rebuild_document_index(database_chunk.document_id, [database_chunk])
    engine = HybridRetrievalEngine(
        vector_store, provider, bm25, rrf_constant=60, candidate_multiplier=3
    )

    result = engine.retrieve("feline cat behavior", RetrievalFilters(), top_k=2)

    assert result.candidates[0].chunk.id == cat_chunk.id
    assert result.candidates[0].semantic_score is not None
    assert result.candidates[0].keyword_score is not None
    assert 0 <= result.confidence_score <= 1
    assert result.semantic_latency_ms >= 0
    assert result.keyword_latency_ms >= 0
    assert result.total_latency_ms >= 0
    assert 0 <= result.candidates[0].normalized_score <= 1

    filtered = engine.retrieve(
        "feline cat behavior", RetrievalFilters(document_ids=(database_chunk.document_id,)), top_k=2
    )

    assert [candidate.chunk.document_id for candidate in filtered.candidates] == [
        database_chunk.document_id
    ]


def test_bm25_returns_normalized_scores_and_honors_page_filter() -> None:
    bm25 = BM25IndexManager()
    timestamp = datetime.now(UTC)
    first = ChunkRecord(
        id="doc:p1:c1",
        document_id="doc",
        filename="source.pdf",
        page_number=1,
        sequence=1,
        text="keyword retrieval uses exact terms",
        upload_timestamp=timestamp,
    )
    second = ChunkRecord(
        id="doc:p2:c1",
        document_id="doc",
        filename="source.pdf",
        page_number=2,
        sequence=1,
        text="keyword keyword retrieval has stronger lexical match",
        upload_timestamp=timestamp,
    )
    bm25.rebuild_document_index("doc", [first, second])

    results = bm25.search("keyword retrieval", RetrievalFilters(page_numbers=(2,)), limit=5)

    assert len(results) == 1
    assert results[0].chunk.id == second.id
    assert results[0].normalized_score == 1.0
