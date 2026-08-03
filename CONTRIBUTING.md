# Contributing

Thanks for helping improve Adaptive Agentic RAG. The project values small, testable changes and local-first operation.

## Development workflow

1. Fork or branch from `master`.
2. Create a Python 3.11-3.13 environment and install `pip install -e '.[dev,evaluation]'`.
3. For UI work, run `npm install` and `npm run dev` in `frontend/`.
4. Add or update tests for behavior changes.
5. Run `pytest`, `ruff check backend evaluation`, and `npm run build` before opening a pull request.

## Design rules

- Keep domain and application services independent from FastAPI and provider SDKs.
- Do not introduce paid APIs, hosted vector databases, or hidden network calls.
- Preserve provenance (`document_id`, filename, page, and chunk ID) through every layer.
- Keep retrieval, generation, and verification independently testable.
- Use typed schemas at API boundaries and structured logs for operational events.

## Pull requests

Describe the problem, the design, tests run, configuration changes, and any migration or resource implications. Do not commit secrets, model weights, databases, uploaded documents, build outputs, or personal data.
