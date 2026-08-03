# API Reference — Milestone 3

## `GET /api/v1/health`

Liveness endpoint for Docker, load balancers, and operators. It returns the service name, version, UTC timestamp, and `status: "ok"`.

## `GET /api/v1/system/info`

Returns safe deployment metadata: application name, version, and environment. It contains no secrets.

## `POST /api/v1/documents/upload`

Accepts `multipart/form-data` with one or more `files` fields. This directly supports browser file-picker and drag-and-drop implementations. Each unique PDF receives a document ID and background indexing job ID. Content-identical PDFs are detected with SHA-256 and return the existing document without re-indexing.
The response reports any individually rejected files while still scheduling accepted files.

## `GET /api/v1/documents/{document_id}`

Returns document metadata, indexing state, page/chunk counts, upload time, and any processing error.

## `GET /api/v1/jobs/{job_id}` and `GET /api/v1/jobs?document_id=...`

Return persisted indexing state, stage, completion percentage, and failure detail for frontend polling.

## `POST /api/v1/retrieval/search`

Runs semantic ChromaDB and BM25 keyword retrieval in parallel. The request supports `query`, optional `top_k`, and document/filename/page filters. The response contains only retrieved passages, normalized channel scores, RRF scores, confidence, and latency; it does not call an LLM or generate an answer.

## Error contract

All handled errors share this shape:

```json
{
  "error": {
    "code": "http_error",
    "message": "Not Found",
    "details": null,
    "request_id": "..."
  }
}
```
