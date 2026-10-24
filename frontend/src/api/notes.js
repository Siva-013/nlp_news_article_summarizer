import { apiFetch } from './client.js';

/** GET /api/v1/notes */
export function listNotes({ limit = 20, offset = 0, term, article_history_id } = {}) {
    const params = new URLSearchParams({ limit, offset });
    if (term) params.set('term', term);
    if (article_history_id) params.set('article_history_id', article_history_id);
    return apiFetch(`/api/v1/notes?${params}`);
}

/** GET /api/v1/notes/search?q= */
export function searchNotes({ q, limit = 20, offset = 0 }) {
    const params = new URLSearchParams({ q, limit, offset });
    return apiFetch(`/api/v1/notes/search?${params}`);
}

/** GET /api/v1/notes/{id} */
export function getNote(id) {
    return apiFetch(`/api/v1/notes/${id}`);
}

/** POST /api/v1/notes */
export function createNote({ content, term, example_sentence, article_history_id }) {
    return apiFetch('/api/v1/notes', {
        method: 'POST',
        body: JSON.stringify({ content, term, example_sentence, article_history_id }),
    });
}

/** PUT /api/v1/notes/{id} */
export function updateNote(id, { content, term, example_sentence, article_history_id, clear_term, clear_example, clear_article_history }) {
    return apiFetch(`/api/v1/notes/${id}`, {
        method: 'PUT',
        body: JSON.stringify({ content, term, example_sentence, article_history_id, clear_term, clear_example, clear_article_history }),
    });
}

/** DELETE /api/v1/notes/{id} */
export function deleteNote(id) {
    return apiFetch(`/api/v1/notes/${id}`, { method: 'DELETE' });
}
