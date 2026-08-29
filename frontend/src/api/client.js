// Central place for every network call the app makes. Components never
// call fetch() directly — they import from here.

export const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function handleJson(res) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const message = data.detail || `Request failed (${res.status})`;
    throw new Error(message);
  }
  return data;
}

export function imageUrl(path) {
  if (!path) return "";
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  return handleJson(res);
}

export async function analyzeImage(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: formData,
  });
  return handleJson(res);
}

export async function listAnalyses(limit = 25, offset = 0) {
  const res = await fetch(`${API_BASE}/api/analyses?limit=${limit}&offset=${offset}`);
  return handleJson(res);
}

export async function getAnalysis(id) {
  const res = await fetch(`${API_BASE}/api/analyses/${id}`);
  return handleJson(res);
}