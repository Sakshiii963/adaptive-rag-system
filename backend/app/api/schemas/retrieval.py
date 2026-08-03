"""Hybrid retrieval request and response contracts; deliberately excludes generation fields."""

from pydantic import BaseModel, Field, field_validator


class RetrievalFilterRequest(BaseModel):
    """Optional document-provenance filtering for both retrieval strategies."""

    document_ids: list[str] = Field(default_factory=list)
    filenames: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)


class RetrievalRequest(BaseModel):
    """A query and optional retrieval controls without any answer-generation behavior."""

    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    filters: RetrievalFilterRequest = Field(default_factory=RetrievalFilterRequest)

    @field_validator("query")
    @classmethod
    def query_cannot_be_blank(cls, value: str) -> str:
        """Reject whitespace-only queries before model embedding work begins."""
        if not value.strip():
            raise ValueError("Query must not be blank.")
        return value


class RetrievedChunkResponse(BaseModel):
    """A source passage with normalized per-channel and fused scores."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    sequence: int
    text: str
    semantic_score: float | None
    keyword_score: float | None
    rrf_score: float
    normalized_score: float


class RetrievalLatencyResponse(BaseModel):
    """Timing breakdown used to observe retrieval performance."""

    semantic_ms: float
    keyword_ms: float
    total_ms: float


class RetrievalResponse(BaseModel):
    """Hybrid retrieval output containing evidence only, never an LLM answer."""

    query: str
    confidence_score: float = Field(ge=0, le=1)
    results: list[RetrievedChunkResponse]
    latency: RetrievalLatencyResponse
