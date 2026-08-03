# Resume assets

## Project summary

Adaptive Agentic RAG is a production-oriented, local-first PDF knowledge platform that combines hybrid retrieval, adaptive query planning, grounded generation, and independent citation verification. It exposes a FastAPI service and a responsive Next.js workspace, with Docker deployment and reproducible evaluation tooling.

## ATS-friendly bullets

- Built a local-first agentic RAG platform using FastAPI, LangGraph, ChromaDB, BM25, BGE embeddings, cross-encoder reranking, Ollama, and Qwen2.5.
- Implemented parallel semantic/keyword retrieval with Reciprocal Rank Fusion, normalized scores, provenance-preserving reranking, confidence gating, deterministic query rewrites, and bounded retries.
- Designed evidence-only generation and an independent verification pipeline that extracts atomic claims, validates citations with semantic support scoring, repairs weak evidence, and prevents unsupported answers.
- Delivered PDF ingestion with PyMuPDF, SHA-256 duplicate detection, configurable chunking, SQLite metadata, background indexing jobs, progress APIs, and incremental persistence.
- Built a typed Next.js/Tailwind frontend for upload progress, grounded chat, evidence inspection, confidence dashboards, verification reports, and developer traces.
- Added Docker Compose deployment, structured request logging, API contracts, unit/integration tests, RAGAS-compatible evaluation, latency benchmarking, and open-source project documentation.

## Interview explanation

The system separates ingestion, retrieval, reranking, planning, generation, and verification behind explicit ports. A question first runs semantic and BM25 retrieval in parallel, fuses candidates with RRF, and reranks them with a cached cross-encoder. LangGraph evaluates confidence and can deterministically rewrite the query within a retry budget. Only successful evidence reaches the versioned Ollama prompt. The verifier extracts claims and checks cited chunks; unsupported material is removed or repaired before returning the response.

## Technical highlights

- Local-only inference and storage; no paid APIs.
- Strong provenance from PDF page through final citation.
- Typed Pydantic/FastAPI contracts and TypeScript API models.
- Bounded loops, explicit insufficient-evidence state, and structured traces.
- Operational readiness through healthchecks, request IDs, durable persistence, and benchmark reports.
