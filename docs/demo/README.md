# Demo assets

This directory contains small synthetic PDFs and a guided demo flow. The content is intentionally fictional and safe to commit; it exists to make local demonstrations repeatable without distributing private documents.

## Demo flow

1. Start Ollama with `ollama serve` and pull `qwen2.5:7b`.
2. Start the backend and frontend using the root quick-start instructions.
3. Drop both PDFs into the workspace and wait for indexing to complete.
4. Ask: `What is the retention period for audit records?`
5. Open the evidence viewer to inspect page and chunk provenance.
6. Enable Developer mode and review retrieval, confidence, generation, and verification diagnostics.
7. Ask: `Which policy describes incident escalation?` and compare the retrieved evidence.

## Example queries

See `example_queries.json`. The synthetic PDFs are `adaptive-retrieval-brief.pdf` and `verification-policy.pdf`.

`screenshots/` contains placeholders for contributors to replace with real, locally captured UI screenshots.
