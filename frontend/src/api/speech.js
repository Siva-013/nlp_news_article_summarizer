import { apiFetch } from './client.js';

/** POST /api/v1/speech/synthesize */
export function synthesizeSpeech({ text, rate = 175, volume = 1.0, voice_id }) {
  const payload = { text, rate, volume };
  if (voice_id) payload.voice_id = voice_id;
  return apiFetch('/api/v1/speech/synthesize', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

/** GET /api/v1/speech/voices */
export function getVoices() {
  return apiFetch('/api/v1/speech/voices');
}

/**
 * Fetch a synthesized WAV with the auth header attached and return a blob
 * object URL for <audio> playback / download links.
 *
 * The audio endpoint requires a Bearer token, and media elements cannot send
 * Authorization headers — so direct URLs get a 401. Fetching the bytes here
 * (authenticated) and playing from a blob URL keeps the endpoint protected.
 */
export async function getAudioBlobUrl(audioId) {
  const res = await apiFetch(`/api/v1/speech/audio/${encodeURIComponent(audioId)}`);
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}
