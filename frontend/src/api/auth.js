import { apiFetch } from './client.js';

/** POST /api/v1/auth/register */
export function register({ username, password }) {
  return apiFetch('/api/v1/auth/register', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
    skipAuth: true,
  });
}

/** POST /api/v1/auth/login */
export function login({ username, password }) {
  return apiFetch('/api/v1/auth/login', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
    skipAuth: true,
  });
}

/** GET /api/v1/auth/me */
export function getMe() {
  return apiFetch('/api/v1/auth/me');
}
