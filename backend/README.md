# Backend

The backend is a FastAPI application rooted at `backend.app.main:app`. It owns the complete local pipeline: PDF ingestion and indexing, hybrid retrieval, reranking, adaptive planning, grounded Ollama generation, and independent citation verification.

## Run

From the repository root:

```bash
pip install -e '.[dev,evaluation]'
uvicorn backend.app.main:app --reload
```

The versioned API is under `/api/v1`; interactive docs are available when `DEBUG=true`. Use `pytest` for the unit/integration suite and `ruff check backend` for linting.

## Boundaries

- `api/`: HTTP routes and Pydantic contracts.
- `services/`: ingestion, retrieval, reranking, and application orchestration.
- `agent/`: typed bounded LangGraph retrieval planner.
- `generation/`: context packing and strict evidence-only prompt/service.
- `verification/`: deterministic claims/citations and semantic support verification.
- `infrastructure/`: Chroma, BM25, embeddings, cross-encoder, SQLite, and Ollama adapters.
- `domain/`: provenance-preserving dataclasses shared across layers.
