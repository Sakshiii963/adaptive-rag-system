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
        normalized = re.sub(r"\n+\s*(?=\[[\d,\s]+\])", " ", answer.strip())
        trailing_markers_match = re.search(r"((?:\[[\d,\s]+\]\s*)+)$", normalized)
        if trailing_markers_match:
            marker_strings = re.findall(
                r"\[([\d,\s]+)\]",
                trailing_markers_match.group(1),
            )

            trailing_markers = tuple(
                int(x.strip())
                for group in marker_strings
                for x in group.split(",")
                if x.strip().isdigit()
            )
        else:
            trailing_markers = ()
        trailing_suffix_start = trailing_markers_match.start() if trailing_markers_match else None
        content = (
            normalized[:trailing_suffix_start].rstrip()
            if trailing_suffix_start is not None
            else normalized
        )
        # Also support the natural `claim. [1]` style while keeping the
        # citation attached to the sentence when another claim follows.
        normalized = re.sub(
            r"([.!?])\s*((?:\[[\d,\s]+\]\s*)+)([.!?]?)",
            lambda match: f" {match.group(2).strip()}{match.group(3) or match.group(1)} ",
            content,
        )
        units = _split_units(normalized)
        claims: list[AtomicClaim] = []
        for raw in units:
            original = raw.strip().lstrip("-•* ").strip()
            if not original or original.lower() == "insufficient evidence.":
                continue
            marker_strings = re.findall(r"\[([\d,\s]+)\]", original)

            markers = tuple(
                int(x.strip())
                for group in marker_strings
                for x in group.split(",")
                if x.strip().isdigit()
            )
            text = re.sub(r"\s*\[[\d,\s]+\]", "", original).strip()
            if text:
                claims.append(AtomicClaim(len(claims) + 1, text, original, markers))
        # A summary paragraph may place its shared source markers once at the
        # end (`Sentence one. Sentence two. [1] [2]`). Treat this as citation
        # scope, then let semantic verification independently validate every
        # inherited claim. Hallucinated sentences therefore remain rejected.
        # A single trailing marker set can scope one preceding claim. When the
        # number of trailing markers equals the number of uncited claims, map in
        # sentence order (the deterministic convention used by the prompt).
        # Otherwise leave claims uncited rather than guessing a broader scope.
        uncited = [c for c in claims if not c.citation_markers]

        if trailing_markers and (
            len(uncited) == 1
            or len(trailing_markers) == 1
            or len(uncited) == len(trailing_markers)
        ):
            marker_groups = (
                [trailing_markers] * len(uncited)
                if len(uncited) == 1 or len(trailing_markers) == 1
                else [(marker,) for marker in trailing_markers]
            )
            assignments = dict(zip((id(claim) for claim in uncited), marker_groups, strict=True))
            claims = [
                replace(
                    claim,
                    original_text=f"{claim.original_text} {' '.join(f'[{m}]' for m in assignments[id(claim)])}",
                    citation_markers=assignments[id(claim)],
                )
                if id(claim) in assignments
                else claim
                for claim in claims
            ]
        return claims


_ABBREVIATIONS = {
    "a.m",
    "approx",
    "dr",
    "e.g",
    "etc",
    "fig",
    "i.e",
    "inc",
    "mr",
    "mrs",
    "ms",
    "no",
    "p.m",
    "prof",
    "st",
    "vs",
}


def _split_units(text: str) -> list[str]:
    """Split claims without treating abbreviations, decimals, or initials as stops."""
    units: list[str] = []
    start = 0
    index = 0
    length = len(text)
    while index < length:
        character = text[index]
        if character == "\n" or character == ";":
            _append_unit(units, text[start:index])
            start = index + 1
            index += 1
            continue
        if character not in ".!?" or not _is_sentence_boundary(text, index):
            index += 1
            continue
        end = index + 1
        while end < length and text[end] in "\"'”’)]}":
            end += 1
        while end < length and text[end].isspace():
            end += 1
        if end >= length or text[end].isupper() or text[end].isdigit() or text[end] in "\"'“‘(-•*":
            _append_unit(units, text[start:end if end < length else index + 1])
            start = end
        index = end if end > index + 1 else index + 1
    _append_unit(units, text[start:])
    return units


def _is_sentence_boundary(text: str, index: int) -> bool:
    """Return whether punctuation at ``index`` ends a claim."""
    if text[index] in "!?":
        return True
    if index > 0 and index + 1 < len(text) and text[index - 1].isdigit() and text[index + 1].isdigit():
        return False
    token_match = re.search(r"([A-Za-z]+(?:\.[A-Za-z]+)*)$", text[:index])
    token = token_match.group(1).lower() if token_match else ""
    if token in _ABBREVIATIONS:
        return False
    next_index = index + 1
    while next_index < len(text) and text[next_index] in "\"'”’)]}":
        next_index += 1
    while next_index < len(text) and text[next_index].isspace():
        next_index += 1
    if len(token) == 1 and token.isalpha() and next_index < len(text):
        return False
    return next_index >= len(text) or text[next_index].isupper() or text[next_index].isdigit()


def _append_unit(units: list[str], value: str) -> None:
    """Append a non-empty splitter unit."""
    if value.strip():
        units.append(value)
