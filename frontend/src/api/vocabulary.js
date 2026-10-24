import { apiFetch } from './client.js';

/**
 * POST /api/v1/vocabulary/extract
 */
export function extractVocabulary({ text, article_id = 'raw_article', max_items = 30, include_entities = true }) {
    return apiFetch('/api/v1/vocabulary/extract', {
        method: 'POST',
        body: JSON.stringify({ text, article_id, max_items, include_entities }),
    });
}

/**
 * POST /api/v1/vocabulary/detect-new
 */
export function detectNewWords({ text, extraction_record, known_vocabulary = [], max_items = 30, include_entities = true }) {
    return apiFetch('/api/v1/vocabulary/detect-new', {
        method: 'POST',
        body: JSON.stringify({ text, extraction_record, known_vocabulary, max_items, include_entities }),
    });
}
