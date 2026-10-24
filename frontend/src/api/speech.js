import { apiFetch, getApiBase } from './client.js';

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

/** Absolute URL for playback / download of a synthesized WAV file */
export function getAudioUrl(audioId) {
  return `${getApiBase()}/api/v1/speech/audio/${encodeURIComponent(audioId)}`;
}
