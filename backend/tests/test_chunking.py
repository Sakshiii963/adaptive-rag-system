"""Unit tests for semantic chunk construction."""

from datetime import UTC, datetime

from backend.app.services.chunking import SemanticChunker


def test_chunker_preserves_page_and_overlap_metadata() -> None:
    chunker = SemanticChunker(chunk_size=60, chunk_overlap=35)
    chunks = chunker.chunk_page(
        document_id="doc-1",
        filename="manual.pdf",
        page_number=3,
        text="First sentence is concise. Second sentence carries context. Third sentence closes.",
        upload_timestamp=datetime.now(UTC),
    )

    assert len(chunks) >= 2
    assert all(chunk.document_id == "doc-1" for chunk in chunks)
    assert all(chunk.filename == "manual.pdf" for chunk in chunks)
    assert all(chunk.page_number == 3 for chunk in chunks)
    assert chunks[0].id == "doc-1:p3:c1"
    assert "Second sentence" in chunks[0].text
    assert "Second sentence" in chunks[1].text
