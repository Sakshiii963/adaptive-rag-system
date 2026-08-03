# Operations — Milestone 1

Structured logs are written to standard output. Each completed request contains `request_id`, method, path, HTTP status, and latency in milliseconds. Clients may provide `X-Request-ID`; otherwise the service generates one and returns it in the response.

Use `/api/v1/health` for liveness. Dependency readiness is intentionally deferred because external RAG dependencies are not part of this milestone.

Uploaded PDFs are saved under `UPLOAD_DIRECTORY`; their SHA-256 is unique in SQLite. Chroma data persists under `CHROMA_PERSIST_DIRECTORY`, while document/job/chunk metadata persists in the SQLite file configured by `METADATA_DATABASE_URL`. Index retries replace vectors and chunks for the document, avoiding duplicates.
