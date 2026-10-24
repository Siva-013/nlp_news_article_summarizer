import { apiFetch } from './client.js';

/** GET /api/v1/history */
export function listHistory({ limit = 50, offset = 0, method } = {}) {
    const params = new URLSearchParams({ limit, offset });
    if (method) params.set('method', method);
    return apiFetch(`/api/v1/history?${params}`);
}

/** GET /api/v1/history/search?q= */
export function searchHistory({ q, limit = 50, offset = 0 }) {
    const params = new URLSearchParams({ q, limit, offset });
    return apiFetch(`/api/v1/history/search?${params}`);
}

/** GET /api/v1/history/{id} */
export function getHistory(id) {
    return apiFetch(`/api/v1/history/${id}`);
}

/** POST /api/v1/history */
export function createHistory({ summary, method, title, article_text, source_article_id, metadata }) {
    return apiFetch('/api/v1/history', {
        method: 'POST',
        body: JSON.stringify({ summary, method, title, article_text, source_article_id, metadata }),
    });
}

/** DELETE /api/v1/history/{id} */
export function deleteHistory(id) {
    return apiFetch(`/api/v1/history/${id}`, { method: 'DELETE' });
}
