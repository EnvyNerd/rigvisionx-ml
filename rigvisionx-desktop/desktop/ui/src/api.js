const DEFAULT_API_BASE = "http://127.0.0.1:8081";
const DEV_PROXY_BASE = "/api";

function looksLikeJwt(value) {
  const token = value.trim();
  return /^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/.test(token);
}

export function normalizeBase(value) {
  if (!value) return "";
  const trimmed = value.trim();
  if (trimmed === "/") return "";
  return trimmed.replace(/\/+$/, "");
}

export function getDefaultApiBase() {
  if (import.meta.env.VITE_API_BASE) {
    return normalizeBase(import.meta.env.VITE_API_BASE);
  }
  if (import.meta.env.DEV) {
    return DEV_PROXY_BASE;
  }
  return DEFAULT_API_BASE;
}

export function buildUrl(base, path) {
  const safeBase = normalizeBase(base);
  const safePath = path.startsWith("/") ? path : `/${path}`;
  if (!safeBase) return safePath;
  return `${safeBase}${safePath}`;
}

export async function readJson(response) {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

export function buildFormData(fields, file) {
  const form = new FormData();
  Object.entries(fields).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") return;
    form.append(key, String(value));
  });
  if (file) {
    form.append("file", file);
  }
  return form;
}

export function buildHeaders(apiKey) {
  if (!apiKey) return undefined;
  const value = apiKey.trim();
  if (!value) return undefined;
  if (value.toLowerCase().startsWith("bearer ")) {
    return { Authorization: value };
  }
  if (looksLikeJwt(value)) {
    return { Authorization: `Bearer ${value}` };
  }
  return { "x-api-key": value };
}
