import type { VerificationResponse } from "../lib/types";

function Metric({ label, value, color = "bg-teal-600" }: { label: string; value: number; color?: string }) {
  return <div><div className="mb-1 flex justify-between text-xs"><span className="muted">{label}</span><span className="font-semibold">{Math.round(value * 100)}%</span></div><div className="h-1.5 rounded-full bg-slate-200 dark:bg-slate-700"><div className={`h-full rounded-full ${color}`} style={{ width: `${Math.max(3, value * 100)}%` }} /></div></div>;
}

export function ConfidenceDashboard({ response }: { response: VerificationResponse }) {
  return <section className="surface rounded-2xl border p-4"><div className="mb-4 flex items-center justify-between"><h3 className="text-sm font-semibold">Confidence dashboard</h3><span className={`rounded-full px-2 py-1 text-[11px] font-semibold ${response.status === "insufficient_evidence" ? "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-200" : "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200"}`}>{response.status.replace("_", " ")}</span></div><div className="space-y-3"><Metric label="Generation confidence" value={response.confidence_score} /><Metric label="Grounding score" value={response.grounding_score} color="bg-indigo-600" /><Metric label="Citation coverage" value={response.citation_coverage_score} color="bg-violet-600" /></div><p className="muted mt-4 text-xs">Verification retries: <span className="font-semibold text-[var(--ink)]">{response.report.retry_count}</span></p></section>;
}
