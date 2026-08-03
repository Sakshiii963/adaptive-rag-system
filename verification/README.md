# Verification module

Milestone 7 is implemented in `backend/app/verification/`. It extracts atomic claims, parses and validates citation markers, scores claim-to-evidence semantic support with the local cross-encoder, calculates citation/grounding scores, and performs bounded targeted retrieval repair through injected callbacks. It is independent of the generation implementation and does not perform frontend or evaluation work.
