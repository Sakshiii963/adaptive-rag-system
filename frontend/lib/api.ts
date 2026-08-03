import type { Job, UploadedDocument, VerificationResponse } from "./types";

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.error?.message ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export function uploadDocuments(files: File[], onProgress: (value: number) => void): Promise<{ documents: UploadedDocument[]; rejected: { filename: string; message: string }[] }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/documents/upload`);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(Math.round((event.loaded / event.total) * 100));
    };
    xhr.onerror = () => reject(new Error("Could not reach the backend."));
    xhr.onload = () => {
      try {
        const body = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) resolve(body);
        else reject(new Error(body?.error?.message ?? "Upload failed."));
      } catch { reject(new Error("The backend returned an invalid response.")); }
    };
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    xhr.send(form);
  });
}

export function getJob(jobId: string): Promise<Job> {
  return request<Job>(`/jobs/${encodeURIComponent(jobId)}`);
}

export function verifyAnswer(query: string): Promise<VerificationResponse> {
  return request<VerificationResponse>("/verification/answer", {
    method: "POST",
    body: JSON.stringify({ query, top_k: 5 }),
  });
}

export { API_BASE };
