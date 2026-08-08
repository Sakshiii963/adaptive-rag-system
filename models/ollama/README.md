# Ollama model setup

Install Ollama locally, start `ollama serve`, and pull the configured model:

```bash
ollama pull qwen2.5:7b
```

Set `OLLAMA_BASE_URL` and `OLLAMA_MODEL` in `.env` if Ollama is not reachable at the default local/container address. The application uses `stream=false`, temperature zero, and a bounded output budget. Ollama is the only answer-generation provider; retrieval and verification remain local Python components.
