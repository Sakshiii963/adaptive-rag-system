# Reranker module

Milestone 4 is implemented in `backend/app/services/reranking.py` and `backend/app/infrastructure/reranker/cross_encoder.py`. It accepts hybrid candidates, performs cached batched `cross-encoder/ms-marco-MiniLM-L-6-v2` scoring, and returns reordered evidence without generating answers.
