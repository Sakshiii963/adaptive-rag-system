# Adaptive Agentic RAG

Adaptive Agentic RAG is a local-first, open-source knowledge assistant for reliable PDF question answering. It combines hybrid retrieval (ChromaDB + BM25), cross-encoder reranking, bounded deterministic query planning, grounded Ollama/Qwen2.5 generation, and independent citation verification. The system returns evidence and a reasoning trace with every answer so unsupported claims can be detected instead of silently generated.

Everything runs locally. There are no OpenAI, Claude, Gemini, Pinecone, or paid-service dependencies.

## Capabilities

- Multi-file PDF upload with SHA-256 duplicate detection and background indexing.
- Page-aware PyMuPDF extraction, configurable semantic chunking, BGE-small embeddings, Chroma persistence, SQLite metadata, and BM25 indexes.
- Parallel semantic and lexical retrieval fused with Reciprocal Rank Fusion.
- Cached batched `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking.
- Typed LangGraph planner with confidence gating, deterministic rewrites, and bounded retries.
- Evidence-only Qwen2.5 generation through a local Ollama server.
- Atomic claim extraction, citation parsing, semantic support verification, evidence repair, and grounding scores.
- Responsive Next.js workspace with upload progress, evidence, verification, confidence, and developer trace panels.

## Quick start

Requirements: Python 3.11-3.13, Node.js 20+, Docker Desktop (optional), and Ollama (only for answer generation).

```bash
cp .env.example .env
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,evaluation]'
ollama pull qwen2.5:7b
uvicorn backend.app.main:app --reload
```

In a second terminal:

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`; API docs are at `http://localhost:8000/docs` when `DEBUG=true`.

## Docker

```bash
cp .env.example .env
docker compose -f docker/docker-compose.yml up --build
```

The backend is published on port 8000 and the frontend on port 3000. Ollama remains a host-local dependency; on macOS/Linux the default `OLLAMA_BASE_URL=http://host.docker.internal:11434` lets the backend container reach it. For production, pin image tags, provide a secret-managed `.env`, put TLS/reverse proxy in front of the services, and mount durable `database/` and `models/` volumes.

## Evaluation

Install the optional evaluation dependencies and run against an indexed, running API:

```bash
python evaluation/run_evaluation.py --dataset evaluation/datasets/sample_questions.json --workers 2
```

The report includes retrieval precision/recall, API-provided faithfulness and relevancy proxies, optional RAGAS faithfulness/relevancy, mean and p95 latency, and throughput. Add gold `relevant_chunk_ids` to dataset cases for meaningful retrieval scores. Reports are written to `evaluation/reports/`.

## Quality checks

```bash
pytest
ruff check backend evaluation
cd frontend && npm run build
docker compose -f docker/docker-compose.yml config
```

## Documentation

- [Architecture and design decisions](docs/architecture.md)
- [Architecture diagrams](docs/architecture_diagrams.md)
- [API reference](docs/api.md)
- [Installation and operations](docs/operations.md)
- [Evaluation guide](docs/evaluation.md)
- [Folder structure](docs/folder_structure.md)
- [Design decisions and future work](docs/design_decisions.md)
- [Demo script and example queries](docs/demo/README.md)
- [Resume summary](docs/resume.md)
- [Interview preparation](docs/interview_questions.md)
- [Contributing](CONTRIBUTING.md)

## License

Released under the MIT License. See [LICENSE](LICENSE).
