# Adaptive Agentic RAG frontend

The Next.js client is an evidence-first workspace for document upload, grounded questions, retrieval evidence, verification results, and agent traces. It is intentionally a presentation and API-consumption layer; retrieval, generation, and verification remain in the FastAPI service.

## Run locally

```bash
cp .env.example .env.local
npm install
npm run dev
```

Set `NEXT_PUBLIC_API_BASE_URL` to the backend API prefix when the backend is not running at `http://localhost:8000/api/v1`.

## Component architecture

- `app/` contains the Next.js shell, metadata, and global Tailwind styles.
- `components/rag-workspace.tsx` owns UI state, polling, API calls, diagnostics, and responsive layout.
- `components/document-sidebar.tsx` handles multi-file drag-and-drop uploads and indexing progress.
- `components/chat-panel.tsx` provides chat history, suggestions, copy, and clear actions.
- `components/evidence-viewer.tsx`, `confidence-dashboard.tsx`, `verification-panel.tsx`, and `trace-panel.tsx` render the explainability surfaces.
- `lib/api.ts` is the typed API boundary; `lib/types.ts` mirrors backend response contracts.

The UI is keyboard accessible, uses semantic labels and live error regions, supports dark mode, and provides empty/loading/error states. Streaming is deliberately not enabled yet, but the chat API boundary is isolated so it can be added without changing the presentation components.
