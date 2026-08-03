"""Maximum-context management for grounded generation."""

from dataclasses import dataclass

from backend.app.domain.entities import RerankedCandidate


@dataclass(frozen=True, slots=True)
class ContextEvidence:
    """Evidence item assigned a stable prompt citation number."""

    marker: int
    candidate: RerankedCandidate
    text: str


class ContextWindowManager:
    """Packs highest-ranked evidence into a bounded character budget without external context."""

    def __init__(self, max_chars: int) -> None:
        if max_chars < 1000:
            raise ValueError("max_chars must be at least 1000.")
        self.max_chars = max_chars

    def pack(self, candidates: tuple[RerankedCandidate, ...]) -> tuple[ContextEvidence, ...]:
        """Keep ranked evidence order and clip only an oversized final passage."""
        packed: list[ContextEvidence] = []
        used_chars = 0
        for marker, candidate in enumerate(candidates, start=1):
            remaining = self.max_chars - used_chars
            if remaining <= 0:
                break
            text = candidate.candidate.chunk.text.strip()
            if len(text) > remaining:
                text = text[:remaining].rsplit(" ", 1)[0].strip()
            if not text:
                break
            packed.append(ContextEvidence(marker, candidate, text))
            used_chars += len(text)
        return tuple(packed)
