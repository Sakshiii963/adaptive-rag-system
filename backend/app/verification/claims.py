"""Deterministic atomic claim extraction."""

import re
from dataclasses import replace

from backend.app.verification.models import AtomicClaim


class AtomicClaimExtractor:
    """Splits answer prose into sentence-level claims while preserving original citation text."""

    def extract(self, answer: str) -> list[AtomicClaim]:
        """Extract non-empty sentence/semicolon units and their inline citation markers."""
        # Models occasionally render a citation on the line immediately after its
        # claim. Attach citation-only lines before splitting so formatting does not
        # turn a properly cited claim into an uncited one.
        normalized = re.sub(r"\n+\s*(?=\[\d+\])", " ", answer.strip())
        trailing_markers_match = re.search(r"((?:\[\d+\]\s*)+)$", normalized)
        trailing_markers = (
            tuple(int(value) for value in re.findall(r"\[(\d+)\]", trailing_markers_match.group(1)))
            if trailing_markers_match
            else ()
        )
        # Also support the natural `claim. [1]` style while keeping the
        # citation attached to the sentence when another claim follows.
        normalized = re.sub(
            r"([.!?])\s*((?:\[\d+\]\s*)+)",
            lambda match: f" {match.group(2).strip()}{match.group(1)} ",
            normalized,
        )
        units = re.split(r"(?<=[.!?])\s+|;\s+|\n+", normalized)
        claims: list[AtomicClaim] = []
        for raw in units:
            original = raw.strip().lstrip("-•* ").strip()
            if not original or original.lower() == "insufficient evidence.":
                continue
            markers = tuple(int(value) for value in re.findall(r"\[(\d+)\]", original))
            text = re.sub(r"\s*\[\d+\]", "", original).strip()
            if text:
                claims.append(AtomicClaim(len(claims) + 1, text, original, markers))
        # A summary paragraph may place its shared source markers once at the
        # end (`Sentence one. Sentence two. [1] [2]`). Treat this as citation
        # scope, then let semantic verification independently validate every
        # inherited claim. Hallucinated sentences therefore remain rejected.
        if trailing_markers and len(claims) > 1:
            claims = [
                replace(
                    claim,
                    original_text=f"{claim.original_text} {' '.join(f'[{m}]' for m in trailing_markers)}",
                    citation_markers=trailing_markers,
                )
                if not claim.citation_markers
                else claim
                for claim in claims
            ]
        return claims
