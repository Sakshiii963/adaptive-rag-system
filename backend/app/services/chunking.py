"""Page-aware semantic text chunking."""

import re

from backend.app.domain.entities import ChunkRecord


class SemanticChunker:
    """Prefers paragraph and sentence boundaries while enforcing predictable chunk limits."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_page(
        self, document_id: str, filename: str, page_number: int, text: str, upload_timestamp
    ) -> list[ChunkRecord]:
        """Split one page into overlap-aware, citation-safe chunks."""
        units = self._semantic_units(text)
        chunks: list[str] = []
        current: list[str] = []
        current_size = 0
        for unit in units:
            if current and current_size + len(unit) + 1 > self.chunk_size:
                chunks.append(" ".join(current))
                current = self._overlap_units(current)
                current_size = sum(len(part) + 1 for part in current)
            if len(unit) > self.chunk_size:
                if current:
                    chunks.append(" ".join(current))
                    current, current_size = [], 0
                chunks.extend(self._split_long_unit(unit))
                continue
            current.append(unit)
            current_size += len(unit) + 1
        if current:
            chunks.append(" ".join(current))
        return [
            ChunkRecord(
                id=f"{document_id}:p{page_number}:c{sequence}",
                document_id=document_id,
                filename=filename,
                page_number=page_number,
                sequence=sequence,
                text=chunk,
                upload_timestamp=upload_timestamp,
            )
            for sequence, chunk in enumerate(chunks, start=1)
            if chunk.strip()
        ]

    def _semantic_units(self, text: str) -> list[str]:
        paragraphs = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\n\s*\n", text)]
        units: list[str] = []
        for paragraph in paragraphs:
            if paragraph:
                units.extend(
                    part.strip() for part in re.split(r"(?<=[.!?])\s+", paragraph) if part.strip()
                )
        return units

    def _overlap_units(self, units: list[str]) -> list[str]:
        overlap: list[str] = []
        size = 0
        for unit in reversed(units):
            if size + len(unit) > self.chunk_overlap:
                break
            overlap.insert(0, unit)
            size += len(unit) + 1
        return overlap

    def _split_long_unit(self, unit: str) -> list[str]:
        words = unit.split()
        parts: list[str] = []
        current: list[str] = []
        size = 0
        for word in words:
            if current and size + len(word) + 1 > self.chunk_size:
                parts.append(" ".join(current))
                current, size = [], 0
            current.append(word)
            size += len(word) + 1
        if current:
            parts.append(" ".join(current))
        return parts
