"""Citation-reference parsing independent of semantic support decisions."""

import re


class CitationParser:
    """Extracts all inline numeric markers, including references not present in evidence."""

    def parse(self, text: str) -> tuple[int, ...]:
        """Return citation markers in appearance order, preserving duplicates for auditability."""
        return tuple(int(value) for value in re.findall(r"\[(\d+)\]", text))
