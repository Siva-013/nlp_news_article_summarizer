import { apiFetch } from './client.js';

/**
 * POST /api/v1/summarize
 * @param {string} text
 * @param {'frequency'|'tfidf'|'textrank'} method
 * @param {number} num_sentences
 * @param {string} [article_id]
 */
export function summarize({ text, method = 'frequency', num_sentences = 3, article_id = 'custom_article' }) {
    return apiFetch('/api/v1/summarize', {
        method: 'POST',
        body: JSON.stringify({ text, method, num_sentences, article_id }),
    });
}
