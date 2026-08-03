# Agent module

Milestone 5 is implemented in `backend/app/agent/`. `graph.py` defines the LangGraph workflow, `state.py` defines typed graph state, `rewrite.py` provides deterministic local rewriting, and `confidence.py` evaluates evidence quality. The graph never calls an LLM or generates an answer.
