import { useEffect, useRef, useState } from 'react';
import { getAudioBlobUrl, getVoices, synthesizeSpeech } from '../api/speech.js';
import { Button } from '../components/ui/Button.jsx';
import { Field } from '../components/ui/Field.jsx';
import { EmptyState } from '../components/ui/EmptyState.jsx';
import { useToast } from '../components/ui/ToastContext.jsx';
import { useAppState } from '../context/AppStateContext.jsx';
import './Listen.css';

const RATE_PRESETS = [
  { label: 'Slow', value: 120 },
  { label: 'Normal', value: 175 },
  { label: 'Fast', value: 240 },
];

function formatBytes(bytes) {
  if (bytes == null) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function Listen() {
  const toast = useToast();
  const { ttsPendingText, setTtsPendingText, lastSummary } = useAppState();
  const [text, setText] = useState(ttsPendingText || lastSummary?.summary || '');
  const [rate, setRate] = useState(175);
  const [volume, setVolume] = useState(1);
  const [voiceId, setVoiceId] = useState('');
  const [voices, setVoices] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [progress, setProgress] = useState(0);
  const [playing, setPlaying] = useState(false);
  const audioRef = useRef(null);

  useEffect(() => {
    if (ttsPendingText) {
      setText(ttsPendingText);
      setTtsPendingText('');
    }
  }, [ttsPendingText, setTtsPendingText]);

  useEffect(() => {
    getVoices()
      .then((res) => setVoices(res.voices || []))
      .catch(() => setVoices([]));
  }, []);

  // Release the latest blob URL when leaving the page
  useEffect(() => () => {
    if (result?.playUrl?.startsWith('blob:')) URL.revokeObjectURL(result.playUrl);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleSynthesize() {
    const trimmed = text.trim();
    if (!trimmed) {
      setError('Provide text to read.');
      return;
    }
    setError('');
    setLoading(true);
    setResult(null);
    setProgress(0);
    try {
      const res = await synthesizeSpeech({
        text: trimmed,
        rate,
        volume,
        voice_id: voiceId || undefined,
      });
      // Fetch the WAV with auth and play from a blob URL — media elements
      // cannot send the Authorization header themselves.
      const playUrl = await getAudioBlobUrl(res.audio_id);
      setResult((prev) => {
        if (prev?.playUrl?.startsWith('blob:')) URL.revokeObjectURL(prev.playUrl);
        return { ...res, playUrl };
      });
      toast('Audio ready', 'success');
    } catch (err) {
      setError(err.detail || err.message || 'Speech synthesis failed.');
      toast('Speech synthesis failed', 'error');
    } finally {
      setLoading(false);
    }
  }

  function onTimeUpdate() {
    const audio = audioRef.current;
    if (!audio || !audio.duration) return;
    setProgress(audio.currentTime / audio.duration);
  }

  function togglePlay() {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) audio.play();
    else audio.pause();
  }

  const selectedVoice = voices.find((v) => v.id === voiceId);

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <p className="eyebrow">Offline text-to-speech</p>
          <h1 className="page-head__title">Listening <span className="text-gradient">studio</span></h1>
          <p className="page-head__sub">
            Synthesis runs entirely on this machine through the pyttsx3 engine — no cloud voices,
            no network calls. Tune voice, speed, and volume, then play or download the WAV.
          </p>
        </div>
        <div className="page-head__aside mono">
          {voices.length} host {voices.length === 1 ? 'voice' : 'voices'}
        </div>
      </header>

      <div className="listen-grid">
        <section className="card" aria-label="Text to synthesize">
          <div className="card__head">
            <span className="card__title">Script</span>
            <Button variant="ghost" size="sm" disabled={!lastSummary?.summary} onClick={() => setText(lastSummary.summary)}>
              Use last summary
            </Button>
          </div>
          <div className="listen-script">
            <Field
              id="tts-text"
              multiline
              rows={11}
              value={text}
              onChange={(e) => { setText(e.target.value); setError(''); }}
              onClear={() => setText('')}
              maxLength={20000}
              error={error}
            />
          </div>
        </section>

        <section className="card" aria-label="Voice controls">
          <div className="card__head">
            <span className="card__title">Voice &amp; delivery</span>
          </div>
          <div className="listen-board">
            <label className="field__label" htmlFor="tts-voice">Voice</label>
            <div className="listen-voices" role="radiogroup" aria-label="Voice">
              <button
                type="button"
                role="radio"
                aria-checked={voiceId === ''}
                className={`listen-voice ${voiceId === '' ? 'listen-voice--active' : ''}`}
                onClick={() => setVoiceId('')}
              >
                <span className="listen-voice__name">Host default</span>
                <span className="listen-voice__meta mono">pyttsx3</span>
              </button>
              {voices.map((voice) => (
                <button
                  key={voice.id}
                  type="button"
                  role="radio"
                  aria-checked={voiceId === voice.id}
                  className={`listen-voice ${voiceId === voice.id ? 'listen-voice--active' : ''}`}
                  onClick={() => setVoiceId(voice.id)}
                >
                  <span className="listen-voice__name">{voice.name || voice.id}</span>
                  <span className="listen-voice__meta mono">
                    {[voice.gender, Array.isArray(voice.languages) ? voice.languages.join(' ') : voice.languages]
                      .filter(Boolean)
                      .join(' · ') || 'host voice'}
                  </span>
                </button>
              ))}
            </div>

            <div className="listen-slider">
              <div className="listen-slider__head">
                <label className="field__label" htmlFor="tts-rate">Speed</label>
                <span className="mono listen-slider__value">{rate} wpm</span>
              </div>
              <input id="tts-rate" type="range" min={50} max={400} step={5} value={rate} onChange={(e) => setRate(Number(e.target.value))} />
              <div className="listen-presets">
                {RATE_PRESETS.map((p) => (
                  <button key={p.value} type="button" className={`listen-preset ${rate === p.value ? 'listen-preset--active' : ''}`} onClick={() => setRate(p.value)}>
                    {p.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="listen-slider">
              <div className="listen-slider__head">
                <label className="field__label" htmlFor="tts-volume">Volume</label>
                <span className="mono listen-slider__value">{Math.round(volume * 100)}%</span>
              </div>
              <input id="tts-volume" type="range" min={0} max={1} step={0.05} value={volume} onChange={(e) => setVolume(Number(e.target.value))} />
            </div>

            <Button variant="primary" size="lg" loading={loading} disabled={!text.trim()} onClick={handleSynthesize}>
              {loading ? 'Synthesizing…' : 'Generate audio'}
            </Button>
          </div>
        </section>
      </div>

      <section aria-label="Audio output" aria-live="polite">
        {!result && !loading ? (
          <EmptyState
            icon="▶"
            title="No audio yet"
            description="Generate speech from your script — the player appears here once the backend returns an audio file."
          />
        ) : null}

        {loading ? (
          <div className="card card--pad listen-generating">
            <span className="listen-generating__dot" aria-hidden="true" />
            <span>Synthesizing with {selectedVoice?.name || 'the host default voice'} at {rate} wpm…</span>
          </div>
        ) : null}

        {result ? (
          <div className="card listen-player">
            <div className="listen-player__main">
              <button
                type="button"
                className="listen-player__play"
                onClick={togglePlay}
                aria-label={playing ? 'Pause' : 'Play'}
              >
                {playing ? '❚❚' : '▶'}
              </button>
              <div className="listen-player__info">
                <div className="meter listen-player__meter">
                  <div className="meter__fill" style={{ width: `${Math.round(progress * 100)}%` }} />
                </div>
                <div className="listen-player__meta mono">
                  <span>{result.file_name || `${result.audio_id}.wav`}</span>
                  <span>{formatBytes(result.file_size_bytes)}</span>
                  <span>{result.rate} wpm</span>
                  <span>vol {Math.round((result.volume ?? 1) * 100)}%</span>
                </div>
              </div>
            </div>
            <a className="listen-player__download" href={result.playUrl} download={result.file_name || `${result.audio_id}.wav`}>
              Download WAV
            </a>
            <audio
              ref={audioRef}
              controls
              src={result.playUrl}
              onTimeUpdate={onTimeUpdate}
              onLoadedMetadata={onTimeUpdate}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onEnded={() => { setPlaying(false); setProgress(0); }}
            >
              Your browser cannot play this WAV file.
            </audio>
          </div>
        ) : null}
      </section>
    </div>
  );
}
