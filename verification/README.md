# Citation verification

`backend/app/verification/` extracts deterministic atomic claims, supports inline/paragraph citations and grouped markers (`[1] [2]`, `[1][2]`, `[1, 2]`), resolves explicit marker-to-reranked-candidate mappings, and scores claim/evidence support with the local cross-encoder. Reports include claim support, invalid citations, coverage, grounding, retry state, and exact structured diagnostics. Unsupported material remains rejected.
