# Hybrid retrieval

The implementation lives in `backend/app/services/retrieval.py` and the Chroma/BM25 adapters under `backend/app/infrastructure/`. Semantic BGE retrieval and BM25 execute in parallel, normalize their channel scores, fuse rankings with Reciprocal Rank Fusion, deduplicate chunk IDs, and preserve document/page/chunk provenance. Filters support document IDs, filenames, and page numbers.
