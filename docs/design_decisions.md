# Design decisions and future work

## Decisions

- **Local-first providers:** Ollama, sentence-transformers, Chroma, BM25, SQLite, and PyMuPDF remove paid API and hosted-data dependencies.
- **Hybrid retrieval:** lexical matching handles exact names and identifiers while dense retrieval handles paraphrases; RRF avoids pretending their raw score scales are comparable.
- **Evidence before generation:** confidence gating ensures the model receives only reranked evidence, and verification remains independent so generation cannot mark its own claims as trusted.
- **Bounded adaptation:** deterministic rewrites and a retry budget make behavior reproducible and prevent agent loops.
- **Provenance as data:** every chunk retains document, filename, page, sequence, and chunk ID through all response models.
- **Durable simple storage:** SQLite and Chroma are appropriate for a local showcase while clean interfaces leave room for distributed replacements.

## Future work

- Add authentication, tenant isolation, document deletion, and encrypted-at-rest storage.
- Add streaming transport after the non-streaming contracts are stable.
- Add a browser-based evaluation dashboard and scheduled regression runs.
- Add hybrid index versioning, document update/delete workflows, and a distributed job queue.
- Add multilingual embedding/reranking profiles and configurable model registries.
- Add OpenTelemetry traces, metrics export, and automated quality gates in CI.
