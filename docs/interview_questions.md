# Interview preparation

## Architecture

1. Why are ingestion, retrieval, generation, and verification separate modules?
2. How does provenance move from a PDF page to an inline citation?
3. What happens when a dependency such as Ollama is unavailable?
4. Which boundaries would become services first at scale?

## Retrieval

1. Why combine BM25 with dense retrieval?
2. Why use RRF instead of directly averaging scores?
3. How are duplicate chunks and metadata filters handled?
4. What do precision, recall, and confidence mean in this system?
5. How would you tune chunk size, overlap, and candidate multiplier?

## LangGraph

1. What state does each graph node read and write?
2. How is the retry loop bounded and prevented from repeating a query?
3. Why is query rewriting deterministic instead of LLM-based here?
4. How would you add a human approval or tool node safely?

## Ollama and Qwen2.5

1. Why keep Ollama behind an infrastructure adapter?
2. How do temperature, context limits, and output budgets affect grounding?
3. How would you warm, cache, or scale local model inference?
4. What is the fallback when the model returns malformed citations?

## RAG and verification

1. Why is retrieval confidence not the same as answer faithfulness?
2. How are atomic claims extracted and cited evidence matched?
3. What does the hallucination guard reject before semantic verification?
4. How should unsupported claims affect the final answer?

## Scaling and operations

1. How would you replace SQLite/Chroma while preserving the ports?
2. How would you queue indexing jobs across workers?
3. Which latency and quality metrics belong on an operational dashboard?
4. How would you handle multi-tenant isolation and document deletion?
5. What are the security risks of accepting arbitrary PDFs and model prompts?
