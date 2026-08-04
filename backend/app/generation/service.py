"""Grounded answer generation over successful adaptive retrieval evidence."""

import re
from typing import Protocol

from backend.app.agent.state import AgentState
from backend.app.core.logging import get_logger
from backend.app.domain.entities import GroundedAnswer, GroundedCitation
from backend.app.generation.context import ContextWindowManager
from backend.app.generation.prompt import GroundedPromptBuilder

logger = get_logger(__name__)


class GenerationProvider(Protocol):
    """Provider port enabling offline tests and future streaming-capable implementations."""

    def generate(self, prompt: str, max_output_tokens: int) -> str:
        """Generate a complete non-streaming answer for one grounded prompt."""


class GroundedGenerationService:
    """Builds strict evidence-only prompts and applies format-level hallucination guards."""

    def __init__(
        self,
        provider: GenerationProvider,
        model_name: str,
        prompt_builder: GroundedPromptBuilder,
        context_manager: ContextWindowManager,
        max_output_tokens: int,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.prompt_builder = prompt_builder
        self.context_manager = context_manager
        self.max_output_tokens = max_output_tokens

    def generate(self, agent_state: AgentState) -> GroundedAnswer:
        """Generate only from an agent state that ended with reranked evidence."""
        candidates = agent_state.get("reranking_result")
        if agent_state.get("status") != "evidence" or not candidates or not candidates.candidates:
            return self._insufficient(agent_state)
        packed = self.context_manager.pack(candidates.candidates)
        if not packed:
            return self._insufficient(agent_state)
        prompt = self.prompt_builder.build(agent_state["original_query"], packed)
        logger.info(
            "generation_prompt_constructed",
            extra={
                "query": agent_state["original_query"],
                "prompt_version": self.prompt_builder.version,
                "evidence_count": len(packed),
                "citation_markers": [item.marker for item in packed],
                "prompt_chars": len(prompt),
            },
        )
        answer = self.provider.generate(prompt, self.max_output_tokens)
        all_markers = _all_citations(answer)
        citations = _parse_citations(answer, len(packed))
        safe = _safe_answer(answer, citations, len(packed))
        logger.info(
            "generation_raw_output",
            extra={
                "raw_output": answer,
                "all_citation_markers": all_markers,
                "valid_citation_markers": citations,
                "safe": safe,
            },
        )
        if not safe:
            return self._insufficient(agent_state)
        coverage = len(set(citations)) / len(packed)
        confidence = round(
            min(1.0, max(0.0, (0.7 * agent_state.get("confidence", 0.0)) + (0.3 * coverage))),
            4,
        )
        return GroundedAnswer(
            answer=answer,
            status="answer",
            confidence_score=confidence,
            evidence_coverage_score=round(coverage, 4),
            citations=tuple(
                GroundedCitation(marker=marker, candidate=packed[marker - 1].candidate)
                for marker in sorted(set(citations))
            ),
            evidence=tuple(item.candidate for item in packed),
            rewritten_query=agent_state.get("rewritten_query"),
            prompt_version=self.prompt_builder.version,
            model_name=self.model_name,
        )

    def generate_batch(self, agent_states: list[AgentState]) -> list[GroundedAnswer]:
        """Provide a batch-friendly service boundary while retaining one grounded state per answer."""
        return [self.generate(agent_state) for agent_state in agent_states]

    def _insufficient(self, agent_state: AgentState) -> GroundedAnswer:
        """Return the only safe answer when the agent did not produce evidence."""
        return GroundedAnswer(
            answer="Insufficient evidence.",
            status="insufficient_evidence",
            confidence_score=0.0,
            evidence_coverage_score=0.0,
            citations=(),
            evidence=(),
            rewritten_query=agent_state.get("rewritten_query"),
            prompt_version=self.prompt_builder.version,
            model_name=self.model_name,
        )


def _parse_citations(answer: str, evidence_count: int) -> list[int]:
    """Parse citation markers without asserting that a cited passage entails a claim."""
    markers = _all_citations(answer)
    return [marker for marker in markers if 1 <= marker <= evidence_count]


def _all_citations(answer: str) -> list[int]:
    """Return every numeric marker before range validation for diagnostics."""
    return [int(value) for value in re.findall(r"\[(\d+)\]", answer)]


def _safe_answer(answer: str, citations: list[int], evidence_count: int) -> bool:
    """Reject empty, uncited, malformed, or explicitly out-of-range citation output."""
    if not answer.strip() or answer.strip().lower() == "insufficient evidence.":
        return False
    all_markers = _all_citations(answer)
    return bool(citations) and len(all_markers) == len(citations) and evidence_count > 0
