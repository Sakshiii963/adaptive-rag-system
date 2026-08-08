# Cross-encoder reranking

`backend/app/services/reranking.py` accepts hybrid candidates and the current query, then performs cached batched `cross-encoder/ms-marco-MiniLM-L-6-v2` inference. The adapter validates score counts, converts non-finite logits to a conservative low score, logs batch telemetry, and preserves every retrieval score and provenance field. It never generates answers.
