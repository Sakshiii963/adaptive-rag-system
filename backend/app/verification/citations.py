"""Citation-reference parsing independent of semantic support decisions."""

import re


class CitationParser:
    """Extracts all inline numeric markers, including references not present in evidence."""

    def parse(self, text: str) -> tuple[int, ...]:
        """Return citation markers in appearance order, preserving duplicates for auditability."""
        markers: list[int] = []
        for group in re.findall(r"\[([\d,\s]+)\]", text):
            markers.extend(int(value) for value in group.split(",") if value.strip().isdigit())
        return tuple(markers)
