"use client";

import { useRef, useState } from "react";
import type { Job, UploadedDocument } from "../lib/types";
import { Icon } from "./icons";

export function DocumentSidebar({ documents, jobs, uploadProgress, onUpload }: { documents: UploadedDocument[]; jobs: Record<string, Job>; uploadProgress: number | null; onUpload: (files: File[]) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const handleFiles = (files: FileList | null) => onUpload(Array.from(files ?? []).filter((file) => file.name.toLowerCase().endsWith(".pdf")));
  return <aside className="surface flex min-h-[680px] w-full max-w-xs flex-col border-r p-5 lg:min-h-screen">
    <div className="mb-8 flex items-center gap-3"><div className="flex h-9 w-9 items-center justify-center rounded-xl bg-teal-700 text-white"><Icon name="spark" /></div><div><p className="text-sm font-semibold">Grounded RAG</p><p className="muted text-xs">Knowledge workspace</p></div></div>
    <div className="mb-5 flex items-center justify-between"><h2 className="text-sm font-semibold">Collections</h2><span className="muted text-xs">{documents.length} docs</span></div>
    <div onDragEnter={() => setDragging(true)} onDragLeave={() => setDragging(false)} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); setDragging(false); handleFiles(event.dataTransfer.files); }} role="button" tabIndex={0} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") inputRef.current?.click(); }} className={`mb-6 rounded-2xl border border-dashed p-5 text-center transition ${dragging ? "border-teal-500 bg-teal-50 dark:bg-teal-950/30" : "line"}`}>
      <Icon name="upload" size={22} /><p className="mt-2 text-sm font-medium">Drop PDFs here</p><p className="muted mt-1 text-xs">or choose multiple files</p><button type="button" onClick={() => inputRef.current?.click()} className="mt-3 rounded-lg bg-teal-700 px-3 py-2 text-xs font-semibold text-white hover:bg-teal-800">Choose files</button><input ref={inputRef} type="file" accept="application/pdf,.pdf" multiple className="sr-only" onChange={(event) => handleFiles(event.target.files)} />
    </div>
    {uploadProgress !== null && <div className="mb-5"><div className="mb-1 flex justify-between text-xs"><span>Uploading PDFs</span><span>{uploadProgress}%</span></div><div className="h-1.5 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700"><div className="h-full bg-teal-600 transition-all" style={{ width: `${uploadProgress}%` }} /></div></div>}
    <div className="scrollbar-thin flex-1 space-y-2 overflow-auto pr-1">
      {documents.length === 0 ? <p className="muted rounded-xl border border-dashed p-4 text-center text-xs">Your collection is empty. Upload a PDF to begin.</p> : documents.map((document) => { const job = document.job_id ? jobs[document.job_id] : undefined; const status = job?.status === "completed" ? "indexed" : job?.status ?? document.status; return <div key={document.document_id} className="rounded-xl border p-3"><div className="flex items-start gap-2"><span className="mt-0.5 text-red-500">PDF</span><div className="min-w-0 flex-1"><p className="truncate text-sm font-medium" title={document.filename}>{document.filename}</p><p className="muted mt-1 text-xs">{status}{job && ` · ${job.progress}%`}</p></div></div>{job && status !== "indexed" && <div className="mt-2 h-1 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700"><div className="h-full bg-teal-500" style={{ width: `${job.progress}%` }} /></div>}</div>; })}
    </div>
    <p className="muted mt-5 text-[11px]">Local-only · Evidence-first retrieval</p>
  </aside>;
}
