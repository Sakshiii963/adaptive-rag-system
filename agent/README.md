# Adaptive retrieval agent

`backend/app/agent/` contains the typed LangGraph planner. It runs hybrid retrieval, cross-encoder reranking, confidence evaluation, deterministic query rewriting, and bounded retries. It returns evidence and a structured trace only; generation and verification are downstream services.

The confidence evaluator records retrieval/reranker components, document coverage, stage mismatches, and multi-document synthesis signals without changing configured thresholds.
