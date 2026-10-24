import { apiFetch } from './client.js';

/** GET /health */
export function checkHealth() {
    return apiFetch('/health');
}
