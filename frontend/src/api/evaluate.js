import { apiFetch } from './client.js';

/** POST /api/v1/evaluate */
export function evaluateSummary({ candidate_summary, reference_summary }) {
    return apiFetch('/api/v1/evaluate', {
        method: 'POST',
        body: JSON.stringify({ candidate_summary, reference_summary }),
    });
}
