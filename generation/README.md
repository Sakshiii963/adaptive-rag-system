# Generation module

Milestone 6 is implemented in `backend/app/generation/` and `backend/app/infrastructure/llm/ollama.py`. It builds a versioned evidence-only prompt, manages context size, calls local Ollama/Qwen2.5, applies citation-format guards, and returns grounded answer metrics without performing citation entailment verification.
