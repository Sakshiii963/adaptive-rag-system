# Installation, operations, and deployment

## Environment variables

`.env.example` is the authoritative template. Important production overrides are `ALLOWED_ORIGINS`, `DEBUG=false`, `OLLAMA_BASE_URL`, model names, upload size, persistence paths, and verification thresholds. Never commit `.env` or model credentials.

## Local development

Run the backend with `uvicorn backend.app.main:app --reload` and the frontend with `cd frontend && npm run dev`. Keep Ollama running separately and pull the configured Qwen2.5 model. The backend lazily loads embedding/reranker models, so the first indexing or reranking request is slower than warm requests.

## Docker development

`docker compose -f docker/docker-compose.yml up --build` starts both services. The backend healthcheck gates frontend startup. Persistent bind mounts keep PDFs, Chroma, SQLite, and model caches outside ephemeral containers.

## Production checklist

- Pin Docker image digests and dependency lockfiles.
- Set `DEBUG=false`, explicit CORS origins, and a non-default public reverse proxy.
- Use durable encrypted storage for `database/` and `models/`.
- Run Ollama on a GPU-capable host when available; restrict its network exposure to the backend.
- Put TLS, request limits, authentication, and audit logging at the edge.
- Back up SQLite metadata and Chroma data together; test restore before upgrades.
- Monitor structured request logs, 4xx/5xx rates, indexing failures, retrieval latency, and verification retry rates.

## Health and troubleshooting

`GET /api/v1/health` is liveness only. A slow first request usually indicates a local model download/load. Indexing failures are persisted on the job and document records; inspect `GET /api/v1/jobs/{job_id}`. All handled errors contain a request ID that can be correlated with structured logs.
