# Folder structure

```text
backend/app/       FastAPI entrypoint, API schemas, domain, services, and adapters
backend/tests/     Unit and integration tests for backend behavior
frontend/          Next.js app, typed API client, and explainability components
evaluation/        RAGAS-compatible datasets, benchmark runner, and report output
database/          Durable Chroma, BM25, SQLite metadata, and upload mount points
docker/            Backend/frontend images and Docker Compose deployment
docs/              Architecture, API, operations, demo, resume, and interview guides
agent/             Public architecture notes for the adaptive planner boundary
retrieval/         Public architecture notes for hybrid retrieval boundaries
reranker/          Public architecture notes for cross-encoder reranking
generation/        Public architecture notes for grounded generation
verification/      Public architecture notes for citation verification
models/            Local model cache mount and Ollama notes
```

The backend follows a ports-and-adapters shape: API routes translate HTTP contracts, application services orchestrate use cases, domain entities carry provenance, and infrastructure adapters own Chroma, BM25, SQLite, cross-encoder, embeddings, and Ollama integrations.
