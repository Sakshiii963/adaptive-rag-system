# API reference

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

## `POST /api/v1/retrieval/rerank`

Runs the existing hybrid retrieval pipeline, then reranks its candidates with the local `cross-encoder/ms-marco-MiniLM-L-6-v2` model. `retrieval_top_k` controls the candidate pool and `top_k` controls the final output. Every original provenance and retrieval score is retained alongside the cross-encoder score, confidence, and latency. No answer is generated.

## `POST /api/v1/agent/run`

Runs the bounded LangGraph adaptive retrieval planner. It executes hybrid retrieval, cross-encoder reranking, confidence evaluation, deterministic query rewriting, and retry transitions until the threshold is met or the retry budget is exhausted. It returns the original/re-written query, status (`evidence` or `insufficient_evidence`), confidence, reasoning steps, structured retrieval trace, and reranked evidence only.

## `POST /api/v1/generation/answer`

Runs the adaptive retrieval agent and sends only successful reranked evidence to local Ollama/Qwen2.5. The response includes the grounded answer, inline citation markers mapped to document/page/chunk provenance, confidence, evidence coverage, prompt version, and the supplied evidence. If the agent does not produce evidence, or the model output violates citation-format guards, it returns `Insufficient evidence.` without using outside context.

## `POST /api/v1/verification/answer`

Runs generation, extracts atomic claims, parses inline citations, verifies every cited passage with the local cross-encoder, and returns a grounding report. Unsupported claims/citations trigger targeted retrieval and regeneration when the configured verification retry budget permits; otherwise unsupported material is removed or the endpoint returns `Insufficient evidence.`.

## Error contract

Citation markers are assigned in prompt order: `[1]` is the first packed reranked chunk and `[2]` the second, across all documents. The verifier audits that explicit mapping and logs expected/actual chunk IDs, document ID, filename, and page. It accepts adjacent, grouped, and paragraph-scoped markers while preserving per-claim semantic checks.

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
