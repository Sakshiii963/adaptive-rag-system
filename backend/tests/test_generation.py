"""Unit and integration coverage for strict evidence-grounded generation."""

from datetime import UTC, datetime

from backend.app.domain.entities import (
    ChunkRecord,
    HybridRetrievalCandidate,
    RerankedCandidate,
    RerankingResult,
)
from backend.app.generation.context import ContextWindowManager
from backend.app.generation.prompt import GroundedPromptBuilder
from backend.app.generation.service import GroundedGenerationService


class FakeGenerationProvider:
    """Offline provider that records the exact prompt supplied to generation."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str, max_output_tokens: int) -> str:
        self.prompts.append(prompt)
        return self.answer


def _agent_state(status: str = "evidence") -> dict:
    timestamp = datetime.now(UTC)
    chunk = ChunkRecord(
        id="doc:p1:c1",
        document_id="doc",
        filename="manual.pdf",
        page_number=1,
        sequence=1,
        text="The system stores documents locally.",
        upload_timestamp=timestamp,
    )
    candidate = HybridRetrievalCandidate(chunk, 0.9, 0.8, 0.02, 0.9)
    reranked = RerankingResult(
        (RerankedCandidate(candidate, 4.0, 0.98, 1),), 0.9, 2.0, "test-reranker", 8
    )
    return {
        "original_query": "Where are documents stored?",
        "rewritten_query": None,
        "status": status,
        "confidence": 0.88,
        "reranking_result": reranked,
    }


def test_grounded_generation_uses_only_numbered_provenance_evidence() -> None:
    provider = FakeGenerationProvider("Documents are stored locally [1].")
    service = GroundedGenerationService(
        provider, "qwen2.5:7b", GroundedPromptBuilder("test-v1"), ContextWindowManager(2000), 128
    )

    answer = service.generate(_agent_state())

    assert answer.status == "answer"
    assert answer.answer == "Documents are stored locally [1]."
    assert answer.citations[0].candidate.candidate.chunk.id == "doc:p1:c1"
    assert answer.evidence_coverage_score == 1.0
    assert answer.prompt_version == "test-v1"
    assert "document_id=doc" in provider.prompts[0]
    assert "Do not use outside knowledge" in provider.prompts[0]
    assert "Where are documents stored?" in provider.prompts[0]


def test_hallucination_guard_rejects_unknown_citations() -> None:
    provider = FakeGenerationProvider("Unsupported claim [99].")
    service = GroundedGenerationService(
        provider, "qwen2.5:7b", GroundedPromptBuilder(), ContextWindowManager(2000), 128
    )

    answer = service.generate(_agent_state())

    assert answer.status == "insufficient_evidence"
    assert answer.answer == "Insufficient evidence."
    assert answer.citations == ()


def test_no_evidence_never_calls_generation_provider() -> None:
    provider = FakeGenerationProvider("This must never be returned.")
    service = GroundedGenerationService(
        provider, "qwen2.5:7b", GroundedPromptBuilder(), ContextWindowManager(2000), 128
    )

    answer = service.generate(_agent_state("insufficient_evidence"))

    assert answer.status == "insufficient_evidence"
    assert answer.answer == "Insufficient evidence."
    assert provider.prompts == []


def test_context_window_clips_and_preserves_ranked_evidence() -> None:
    state = _agent_state()
    candidate = state["reranking_result"].candidates[0]
    candidate = RerankedCandidate(
        candidate.candidate, candidate.reranker_score, candidate.normalized_reranker_score, 1
    )
    packed = ContextWindowManager(1000).pack((candidate,))

    assert packed[0].marker == 1
    assert packed[0].candidate.candidate.chunk.id == "doc:p1:c1"
