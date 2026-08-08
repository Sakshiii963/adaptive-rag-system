# Grounded generation

`backend/app/generation/` packs ranked evidence into a bounded context and builds a versioned evidence-only prompt for local Ollama/Qwen2.5. The service accepts only successful adaptive-agent evidence, requires valid numeric citations, records raw provider output in structured diagnostics, and returns grounded answer metrics. Semantic citation entailment is performed independently by `backend/app/verification/`.
