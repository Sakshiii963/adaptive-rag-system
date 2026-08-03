"use client";

import { useEffect, useState } from "react";
import { getJob, uploadDocuments, verifyAnswer } from "../lib/api";
import type { ChatMessage, Job, UploadedDocument, VerificationResponse } from "../lib/types";
import { ChatPanel } from "./chat-panel";
import { ConfidenceDashboard } from "./confidence-dashboard";
import { DocumentSidebar } from "./document-sidebar";
import { EvidenceViewer } from "./evidence-viewer";
import { Icon } from "./icons";
import { TracePanel } from "./trace-panel";
import { VerificationPanel } from "./verification-panel";

export function RagWorkspace() {
  const [documents, setDocuments] = useState<UploadedDocument[]>([]);
  const [jobs, setJobs] = useState<Record<string, Job>>({});
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [latest, setLatest] = useState<VerificationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [developerMode, setDeveloperMode] = useState(false);
  const [dark, setDark] = useState(false);
  const [diagnostics, setDiagnostics] = useState<{ requestId: string; latencyMs: number } | null>(null);

  useEffect(() => { document.documentElement.classList.toggle("dark", dark); }, [dark]);
  useEffect(() => { const active = Object.values(jobs).some((job) => job.status === "queued" || job.status === "running"); if (!active) return; const timer = window.setTimeout(async () => { const updates = await Promise.all(Object.values(jobs).filter((job) => job.status === "queued" || job.status === "running").map((job) => getJob(job.job_id).catch(() => job))); setJobs((current) => Object.fromEntries(updates.map((job) => [job.job_id, job]))); }, 1500); return () => window.clearTimeout(timer); }, [jobs]);

  const handleUpload = async (files: File[]) => { if (!files.length) return; setError(null); setUploadProgress(0); try { const response = await uploadDocuments(files, setUploadProgress); setDocuments((current) => [...response.documents, ...current.filter((doc) => !response.documents.some((newDoc) => newDoc.document_id === doc.document_id))]); const newJobs = Object.fromEntries(response.documents.filter((doc) => doc.job_id).map((doc) => [doc.job_id, { job_id: doc.job_id!, document_id: doc.document_id, status: "queued", progress: 0, stage: "queued", error_message: null, created_at: doc.upload_timestamp, started_at: null, completed_at: null }])); setJobs((current) => ({ ...current, ...newJobs })); if (response.rejected.length) setError(response.rejected.map((item) => `${item.filename}: ${item.message}`).join(" ")); } catch (uploadError) { setError(uploadError instanceof Error ? uploadError.message : "Upload failed."); } finally { window.setTimeout(() => setUploadProgress(null), 600); } };
  const ask = async (query: string) => { setError(null); setLoading(true); const requestId = crypto.randomUUID(); const started = performance.now(); setMessages((current) => [...current, { id: crypto.randomUUID(), role: "user", text: query, createdAt: new Date().toISOString() }]); try { const response = await verifyAnswer(query); setDiagnostics({ requestId, latencyMs: Math.round(performance.now() - started) }); setLatest(response); setMessages((current) => [...current, { id: crypto.randomUUID(), role: "assistant", text: response.answer, response, createdAt: new Date().toISOString() }]); } catch (requestError) { setDiagnostics({ requestId, latencyMs: Math.round(performance.now() - started) }); setError(requestError instanceof Error ? requestError.message : "Could not verify the answer."); } finally { setLoading(false); } };
  return <main className="min-h-screen bg-[var(--surface)] text-[var(--ink)]"><div className="flex min-h-screen flex-col lg:flex-row"><DocumentSidebar documents={documents} jobs={jobs} uploadProgress={uploadProgress} onUpload={handleUpload} /><div className="flex min-w-0 flex-1 flex-col"><header className="surface flex items-center justify-end gap-2 border-b px-5 py-3"><span className="muted mr-auto text-xs">Local workspace</span><label className="flex cursor-pointer items-center gap-2 text-xs"><input type="checkbox" checked={developerMode} onChange={(event) => setDeveloperMode(event.target.checked)} className="accent-teal-700" />Developer mode</label><button type="button" aria-label="Toggle dark mode" onClick={() => setDark((value) => !value)} className="rounded-lg border p-2 hover:bg-slate-100 dark:hover:bg-slate-800"><Icon name={dark ? "sun" : "moon"} size={16} /></button></header><div className="grid flex-1 gap-6 p-5 xl:grid-cols-[minmax(0,1fr)_360px]"><ChatPanel messages={messages} loading={loading} onAsk={ask} onClear={() => { setMessages([]); setLatest(null); setDiagnostics(null); }} /><aside className="space-y-4">{error && <div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-xs text-rose-800 dark:border-rose-900 dark:bg-rose-950/40 dark:text-rose-200">{error}</div>}{loading && !latest && <div className="surface animate-pulse rounded-2xl border p-5"><div className="h-3 w-2/5 rounded bg-slate-200 dark:bg-slate-800" /><div className="mt-3 h-3 w-4/5 rounded bg-slate-200 dark:bg-slate-800" /><div className="mt-3 h-24 rounded bg-slate-200 dark:bg-slate-800" /></div>}{latest ? <><ConfidenceDashboard response={latest} /><VerificationPanel report={latest.report} /><EvidenceViewer evidence={latest.evidence} /><TracePanel response={latest} developerMode={developerMode} diagnostics={diagnostics} /></> : !loading && <div className="surface rounded-2xl border p-5"><p className="text-sm font-semibold">Evidence & verification</p><p className="muted mt-2 text-xs leading-5">Your answer&apos;s evidence, confidence, claims, and reasoning trace will appear here.</p></div>}</aside></div></div></div></main>;
}
