/**
 * Central API fetch wrapper.
 * Reads VITE_API_BASE_URL from the Vite environment.
 * Attaches the saved JWT (if any) to every request; a 401 response clears
 * the session and notifies the app via the 'lexica:unauthorized' event.
 */

const BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const TOKEN_KEY = 'lexica_token';

export function getApiBase() {
  return BASE.replace(/\/$/, '');
}

export function getStoredToken() {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function storeToken(token) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable — session stays in memory only */
  }
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
  const { headers, skipAuth, ...rest } = options;
  const url = `${getApiBase()}${path}`;

  const authHeaders = {};
  if (!skipAuth) {
    const token = getStoredToken();
    if (token) authHeaders.Authorization = `Bearer ${token}`;
  }

  let res;
  try {
    res = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...authHeaders,
        ...(headers || {}),
      },
      ...rest,
    });
  } catch {
    throw new ApiError(0, 'Network error — is the FastAPI backend running?');
  }

  if (res.status === 401 && !skipAuth) {
    storeToken(null);
    window.dispatchEvent(new CustomEvent('lexica:unauthorized'));
    let detail = 'Session expired. Please sign in again.';
    try {
      const body = await res.json();
      detail = formatDetail(body, detail);
    } catch {
      /* ignore parse error */
    }
    throw new ApiError(401, detail);
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
