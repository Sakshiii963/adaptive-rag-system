# Adaptive Agentic RAG

Local-first foundation for an adaptive, verifiable Retrieval-Augmented Generation system.

## Milestone 2 status

The repository contains the production backend foundation plus local PDF ingestion: multi-file upload, SHA-256 duplicate detection, PyMuPDF page extraction, semantic chunking, BGE embedding integration, Chroma persistence, SQLite metadata, BM25 index construction, and persisted background-job progress. Retrieval, LangGraph query planning, reranking, and answer generation are intentionally not implemented yet.

## Run locally

```bash
cp .env.example .env
docker compose -f docker/docker-compose.yml up --build
```

The API is then available at `http://localhost:8000`; interactive API documentation is at `/docs`.

## Verify

```bash
curl http://localhost:8000/api/v1/health
```

## Upload PDFs

```bash
curl -F "files=@/absolute/path/to/document.pdf" http://localhost:8000/api/v1/documents/upload
```

Submit repeated `files` form fields to upload multiple PDFs in one request. Poll the returned `job_id` at `/api/v1/jobs/{job_id}`.

## Project layout

See `docs/architecture.md` for the Milestone 1 boundaries and run instructions.
