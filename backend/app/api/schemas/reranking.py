"""Cross-encoder reranking request and response contracts."""

from pydantic import BaseModel, Field

from backend.app.api.schemas.retrieval import RetrievalFilterRequest


class RerankingRequest(BaseModel):
    """Runs existing hybrid retrieval and then cross-encoder ordering only."""

    query: str = Field(min_length=1, max_length=2000)
    retrieval_top_k: int | None = Field(default=None, ge=1, le=50)
    top_k: int | None = Field(default=None, ge=1, le=50)
    filters: RetrievalFilterRequest = Field(default_factory=RetrievalFilterRequest)


class RerankedChunkResponse(BaseModel):
    """Reranked passage retaining every hybrid retrieval score and provenance field."""

    rank: int
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    sequence: int
    text: str
    semantic_score: float | None
    keyword_score: float | None
    rrf_score: float
    retrieval_normalized_score: float
    reranker_score: float
    normalized_reranker_score: float


class RerankingLatencyResponse(BaseModel):
    """Retrieval and reranking timing breakdown."""

    retrieval_ms: float
    reranking_ms: float
    total_ms: float


class RerankingResponse(BaseModel):
    """Evidence-only reranked result; no generated answer is included."""

    query: str
    confidence_score: float = Field(ge=0, le=1)
    model_name: str
    batch_size: int
    results: list[RerankedChunkResponse]
    latency: RerankingLatencyResponse
