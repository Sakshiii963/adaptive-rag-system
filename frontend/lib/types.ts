export type DocumentStatus = "queued" | "indexing" | "indexed" | "failed";

export type UploadedDocument = {
  document_id: string;
  filename: string;
  status: DocumentStatus;
  duplicate: boolean;
  job_id: string | null;
  upload_timestamp: string;
};

export type Job = {
  job_id: string;
  document_id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: number;
  stage: string;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
};

export type Evidence = {
  rank: number;
  chunk_id: string;
  document_id: string;
  filename: string;
  page_number: number;
  sequence: number;
  text: string;
  semantic_score: number | null;
  keyword_score: number | null;
  rrf_score: number;
  retrieval_normalized_score: number;
  reranker_score: number;
  normalized_reranker_score: number;
};

export type ClaimVerification = {
  claim_id: number;
  text: string;
  citation_markers: number[];
  support_scores: number[];
  supported: boolean;
  reason: string;
};

export type VerificationReport = {
  claims: ClaimVerification[];
  unsupported_claim_ids: number[];
  unsupported_citations: number[];
  citation_coverage_score: number;
  grounding_score: number;
  retry_triggered: boolean;
  retry_count: number;
};

export type TraceStep = {
  node: string;
  attempt: number;
  timestamp: string;
  duration_ms: number;
  details: Record<string, unknown>;
};

export type VerificationResponse = {
  answer: string;
  status: "verified" | "repaired" | "insufficient_evidence";
  confidence_score: number;
  citation_coverage_score: number;
  grounding_score: number;
  rewritten_query: string | null;
  report: VerificationReport;
  evidence: Evidence[];
  trace: TraceStep[];
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  response?: VerificationResponse;
  createdAt: string;
};
