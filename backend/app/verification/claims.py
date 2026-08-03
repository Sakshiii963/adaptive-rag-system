"""Deterministic atomic claim extraction."""

import re

from backend.app.verification.models import AtomicClaim


class AtomicClaimExtractor:
    """Splits answer prose into sentence-level claims while preserving original citation text."""

    def extract(self, answer: str) -> list[AtomicClaim]:
        """Extract non-empty sentence/semicolon units and their inline citation markers."""
        units = re.split(r"(?<=[.!?])\s+|;\s+|\n+", answer.strip())
        claims: list[AtomicClaim] = []
        for raw in units:
            original = raw.strip().lstrip("-•* ").strip()
            if not original or original.lower() == "insufficient evidence.":
                continue
            markers = tuple(int(value) for value in re.findall(r"\[(\d+)\]", original))
            text = re.sub(r"\s*\[\d+\]", "", original).strip()
            if text:
                claims.append(AtomicClaim(len(claims) + 1, text, original, markers))
        return claims
