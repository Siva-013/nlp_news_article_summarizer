/**
 * Central API fetch wrapper.
 * Reads VITE_API_BASE_URL from the Vite environment.
 */

const BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export function getApiBase() {
  return BASE.replace(/\/$/, '');
}

export class ApiError extends Error {
  constructor(status, detail) {
    super(typeof detail === 'string' ? detail : `HTTP ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

function formatDetail(body, fallback) {
  if (!body) return fallback;
  const detail = body.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => item?.msg || item?.detail || JSON.stringify(item))
      .join('; ');
  }
  if (detail && typeof detail === 'object') {
    return JSON.stringify(detail);
  }
  return fallback;
}

export async function apiFetch(path, options = {}) {
  const { headers, ...rest } = options;
  const url = `${getApiBase()}${path}`;
  let res;
  try {
    res = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...(headers || {}),
      },
      ...rest,
    });
  } catch {
    throw new ApiError(0, 'Network error — is the FastAPI backend running?');
  }

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = formatDetail(body, detail);
    } catch {
      /* ignore parse error */
    }
    throw new ApiError(res.status, detail);
  }

  if (res.status === 204) return null;

  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) {
    return res.json();
  }
  return res;
}
