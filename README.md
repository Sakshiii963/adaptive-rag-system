# Adaptive Agentic RAG

Local-first foundation for an adaptive, verifiable Retrieval-Augmented Generation system.

## Milestone 5 status

The repository contains the production backend foundation, local PDF ingestion, hybrid retrieval, cross-encoder reranking, and a bounded LangGraph adaptive retrieval planner. It performs deterministic local query rewrites, retries weak retrieval, emits structured reasoning traces, and returns reranked evidence only. Answer generation, citation verification, and the frontend are intentionally not implemented yet.

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

## Search indexed documents

```bash
curl -X POST http://localhost:8000/api/v1/retrieval/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"your question", "top_k":5}'
```

## Project layout

See `docs/architecture.md` for the current boundaries and run instructions.
