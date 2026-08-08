"""Versioned strict prompt template for local grounded generation."""

from backend.app.generation.context import ContextEvidence


class GroundedPromptBuilder:
    """Builds a prompt whose only factual context is numbered retrieved evidence."""

    def __init__(self, version: str = "v1") -> None:
        self.version = version

    def build(self, query: str, evidence: tuple[ContextEvidence, ...]) -> str:
        """Render the stable prompt contract, including provenance for every passage."""
        evidence_block = "\n\n".join(
            (
                f"[{item.marker}] document_id={item.candidate.candidate.chunk.document_id} "
                f"filename={item.candidate.candidate.chunk.filename} "
                f"page={item.candidate.candidate.chunk.page_number} "
                f"chunk_id={item.candidate.candidate.chunk.id}\n{item.text}"
            )
            for item in evidence
        )
        return f"""PROMPT_VERSION: {self.version}
You are a grounded knowledge assistant.

You MUST answer ONLY using the supplied evidence.

Rules:

1. Never use outside knowledge. Do not use outside knowledge beyond the supplied evidence.
2. Every factual statement MUST end with one or more citations like [1].
3. Never invent citations.
4. If the evidence does not contain the answer, reply exactly:

Insufficient evidence.

5. If the question asks for:
   - a summary
   - an overview
   - a description
   - a list
   - multiple facts

then FIRST read ALL evidence blocks before writing the answer.

6. Combine information from every relevant evidence block.

7. Do NOT answer using only the first matching chunk.

8. Ignore irrelevant evidence blocks.

9. Return only the answer.

USER QUESTION:

{query}

SUPPLIED EVIDENCE:

{evidence_block}

ANSWER:
"""
