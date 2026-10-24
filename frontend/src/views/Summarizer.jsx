import { useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { summarize } from '../api/summarize.js';
import { createHistory } from '../api/history.js';
import { synthesizeSpeech, getAudioUrl } from '../api/speech.js';
import { Button } from '../components/ui/Button.jsx';
import { Field } from '../components/ui/Field.jsx';
import { SkeletonBlock } from '../components/ui/Skeleton.jsx';
import { EmptyState } from '../components/ui/EmptyState.jsx';
import { StatusBadge } from '../components/ui/StatusBadge.jsx';
import { useToast } from '../components/ui/ToastContext.jsx';
import { useAppState } from '../context/AppStateContext.jsx';
import './Summarizer.css';

const METHODS = [
  { id: 'frequency', name: 'Frequency', blurb: 'Luhn-style normalized word frequency scoring.' },
  { id: 'tfidf', name: 'TF–IDF', blurb: 'Term rarity weighted across the document.' },
  { id: 'textrank', name: 'TextRank', blurb: 'Sentence graph centrality via PageRank.' },
];

const SAMPLE_ARTICLE = `Scientists have developed a new type of battery that can store significantly more energy than current lithium-ion batteries. The research team at MIT published their findings in the journal Nature Energy, detailing how the solid-state battery uses a special ceramic electrolyte instead of liquid. The new design eliminates fire risk while doubling energy density. Commercial applications could emerge within five years, according to project lead Dr. Sarah Chen. Auto manufacturers including Toyota and Ford have already expressed interest in licensing the technology for electric vehicles.`;

function countSentences(text) {
  if (!text.trim()) return 0;
  return text.split(/(?<=[.!?])\s+/).filter((s) => s.trim().length > 0).length;
}

/** Renders top_words for either backend shape (frequency / tfidf). */
function TopWords({ topWords, method }) {
  const rows = useMemo(() => {
    if (!Array.isArray(topWords) || topWords.length === 0) return [];
    if (method === 'tfidf') {
      const max = Math.max(...topWords.map((w) => Number(w.tfidf) || 0), 1e-9);
      return topWords.map((w) => ({ label: w.term, value: Number(w.tfidf) || 0, pct: (Number(w.tfidf) || 0) / max * 100, tag: `df ${w.df ?? '—'}` }));
    }
    const max = Math.max(...topWords.map((w) => Number(w.normalized_frequency) || 0), 1e-9);
    return topWords.map((w) => ({ label: w.word, value: Number(w.frequency) || 0, pct: (Number(w.normalized_frequency) || 0) / max * 100, tag: `×${w.frequency}` }));
  }, [topWords, method]);

  if (rows.length === 0) return null;

  return (
    <details className="sum-card__fold">
      <summary>Top scoring words <span className="sum-card__fold-hint">({method === 'tfidf' ? 'TF–IDF weight' : 'normalized frequency'})</span></summary>
      <ul className="topwords">
        {rows.map((r) => (
          <li key={r.label} className="topwords__row">
            <span className="topwords__label">{r.label}</span>
            <span className="meter topwords__meter"><span className="meter__fill" style={{ width: `${Math.max(3, r.pct)}%` }} /></span>
            <span className="topwords__value mono">{r.value.toFixed(method === 'tfidf' ? 3 : 0)}</span>
            <span className="topwords__tag mono">{r.tag}</span>
          </li>
        ))}
      </ul>
    </details>
  );
}

function SummaryResult({ result, sourceSentenceCount, onListen, audioLoading, audioError, onCopy, copied, navigateActions }) {
  const compression = sourceSentenceCount > 0
    ? Math.round((result.num_sentences_selected / sourceSentenceCount) * 100)
    : null;

  return (
    <article className="card sum-result">
      <div className="card__head">
        <div className="sum-result__badges">
          <StatusBadge tone="accent">{result.method}</StatusBadge>
          <StatusBadge tone="neutral">
            {result.num_sentences_selected}/{result.num_sentences_requested} sentences
          </StatusBadge>
          {compression != null ? <StatusBadge tone="info">{compression}% of source</StatusBadge> : null}
        </div>
      </div>

      <div className="sum-result__body">
        <p className="sum-result__text">{result.summary || '—'}</p>

        {result.selected_sentences?.length > 0 ? (
          <details className="sum-card__fold" open>
            <summary>Extracted sentences &amp; scores</summary>
            <ul className="sum-sentences">
              {result.selected_sentences.map((s) => (
                <li key={s.sentence_index} className="sum-sentences__row">
                  <span className="sum-sentences__idx mono">#{s.sentence_index + 1}</span>
                  <span className="sum-sentences__text">{s.original_text}</span>
                  <span className="sum-sentences__score mono">{Number(s.score).toFixed(4)}</span>
                </li>
              ))}
            </ul>
          </details>
        ) : null}

        <TopWords topWords={result.top_words} method={result.method} />

        <div className="sum-result__actions">
          <Button variant="secondary" size="sm" onClick={onCopy}>{copied ? 'Copied ✓' : 'Copy'}</Button>
          <Button variant="secondary" size="sm" loading={audioLoading} onClick={onListen}>
            {audioLoading ? 'Synthesizing…' : 'Listen ▶'}
          </Button>
          {navigateActions}
        </div>

        {audioError ? <div className="callout-error sum-result__audio-error">{audioError}</div> : null}
      </div>
    </article>
  );
}

export function Summarizer() {
  const [text, setText] = useState('');
  const [method, setMethod] = useState('textrank');
  const [numSentences, setNumSentences] = useState(3);
  const [loading, setLoading] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [result, setResult] = useState(null);
  const [compareResults, setCompareResults] = useState(null);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);
  const [audioLoading, setAudioLoading] = useState(false);
  const [audioError, setAudioError] = useState('');
  const audioRef = useRef(null);
  const navigate = useNavigate();
  const toast = useToast();
  const {
    setLastArticleText,
    setLastSummary,
    setTtsPendingText,
    setPendingNote,
    setLastHistoryId,
  } = useAppState();

  const stats = useMemo(() => ({
    words: text.trim() ? text.trim().split(/\s+/).length : 0,
    chars: text.length,
    sentences: countSentences(text),
  }), [text]);

  async function runSummarize(m, n) {
    const res = await summarize({ text: text.trim(), method: m, num_sentences: n });
    try {
      const saved = await createHistory({
        summary: res.summary,
        method: res.method,
        title: text.trim().split('\n')[0].slice(0, 120),
        article_text: text.trim(),
        metadata: {
          num_sentences_requested: res.num_sentences_requested,
          num_sentences_selected: res.num_sentences_selected,
        },
      });
      if (saved?.id) setLastHistoryId(saved.id);
    } catch {
      /* history save is best-effort */
    }
    return res;
  }

  async function handleAnalyze() {
    if (!text.trim()) {
      setError('Paste or load an article first.');
      return;
    }
    setError('');
    setLoading(true);
    setResult(null);
    setCompareResults(null);
    try {
      const res = await runSummarize(method, numSentences);
      setResult(res);
      setLastArticleText(text.trim());
      setLastSummary(res);
      if (!res.summary) toast('The extractor returned an empty summary for this input', 'warning');
    } catch (err) {
      setError(err.detail || err.message || 'Summarization failed.');
      toast('Summarization failed', 'error');
    } finally {
      setLoading(false);
    }
  }

  async function handleCompare() {
    if (!text.trim()) {
      setError('Paste or load an article first.');
      return;
    }
    setError('');
    setComparing(true);
    setResult(null);
    setCompareResults(null);
    try {
      const settled = await Promise.allSettled(
        METHODS.map((m) => runSummarize(m.id, numSentences)),
      );
      const entries = METHODS.map((m, i) => ({
        method: m,
        result: settled[i].status === 'fulfilled' ? settled[i].value : null,
        error: settled[i].status === 'rejected' ? (settled[i].reason?.detail || settled[i].reason?.message) : null,
      }));
      setCompareResults(entries);
      const ok = entries.find((e) => e.result);
      if (ok) {
        setLastArticleText(text.trim());
        setLastSummary(ok.result);
      }
      if (entries.some((e) => e.error)) toast('One or more methods failed to run', 'warning');
    } finally {
      setComparing(false);
    }
  }

  async function handleCopy() {
    if (!result?.summary) return;
    try {
      await navigator.clipboard.writeText(result.summary);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
      toast('Summary copied to clipboard', 'success');
    } catch {
      toast('Clipboard unavailable', 'error');
    }
  }

  async function handleListen() {
    if (!result?.summary) return;
    setAudioLoading(true);
    setAudioError('');
    try {
      const res = await synthesizeSpeech({ text: result.summary, rate: 175, volume: 1.0 });
      if (audioRef.current) {
        audioRef.current.src = getAudioUrl(res.audio_id);
        audioRef.current.load();
        await audioRef.current.play();
      }
    } catch (err) {
      setAudioError(err.detail || err.message || 'Speech synthesis failed.');
      toast('Speech synthesis failed', 'error');
    } finally {
      setAudioLoading(false);
    }
  }

  function handleKeyDown(event) {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      event.preventDefault();
      if (!loading && text.trim()) handleAnalyze();
    }
  }

  const navigateActions = (
    <>
      <Button variant="ghost" size="sm" onClick={() => { setTtsPendingText(result.summary); navigate('/listen'); }}>
        Listen studio →
      </Button>
      <Button variant="ghost" size="sm" onClick={() => navigate('/vocabulary')}>
        Extract vocabulary →
      </Button>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => {
          setPendingNote({
            content: result.summary,
            term: result.method,
            example_sentence: text.trim().split(/[.!?]/)[0]?.slice(0, 240) || '',
          });
          navigate('/notes');
        }}
      >
        Save as note →
      </Button>
    </>
  );

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <p className="eyebrow">Extractive summarization</p>
          <h1 className="page-head__title">Summarize <span className="text-gradient">news articles</span></h1>
          <p className="page-head__sub">
            Sentences are selected and reordered from the original article — nothing is rewritten.
            Every run is archived to history automatically.
          </p>
        </div>
        <div className="page-head__aside mono">
          ctrl+enter to run
        </div>
      </header>

      <section className="card" aria-label="Article input">
        <div className="card__head">
          <span className="card__title">Article</span>
          <div className="stat-row">
            <span className="stat-chip">words <b>{stats.words}</b></span>
            <span className="stat-chip">sentences <b>{stats.sentences}</b></span>
            <span className="stat-chip">chars <b>{stats.chars}</b></span>
          </div>
        </div>
        <div className="sum-input">
          <Field
            id="article-input"
            multiline
            rows={9}
            value={text}
            onChange={(e) => { setText(e.target.value); if (error) setError(''); }}
            onClear={() => { setText(''); setResult(null); setCompareResults(null); setError(''); }}
            onKeyDown={handleKeyDown}
            placeholder="Paste a news article…"
            maxLength={50000}
            error={error}
          />
          <div className="sum-input__tray">
            <Button variant="ghost" size="sm" onClick={() => { setText(SAMPLE_ARTICLE); setError(''); }}>
              Load sample article
            </Button>
          </div>
        </div>
      </section>

      <section className="sum-controls" aria-label="Algorithm and length settings">
        <div className="sum-methods" role="radiogroup" aria-label="Summarization method">
          {METHODS.map((m) => (
            <button
              type="button"
              key={m.id}
              role="radio"
              aria-checked={method === m.id}
              className={`sum-method ${method === m.id ? 'sum-method--active' : ''}`}
              onClick={() => setMethod(m.id)}
            >
              <span className="sum-method__name">{m.name}</span>
              <span className="sum-method__blurb">{m.blurb}</span>
            </button>
          ))}
        </div>

        <div className="card sum-length">
          <label className="field__label" htmlFor="num-sentences">Summary length</label>
          <div className="sum-stepper">
            <button type="button" className="sum-stepper__btn" onClick={() => setNumSentences((n) => Math.max(1, n - 1))} aria-label="Fewer sentences">−</button>
            <output id="num-sentences" className="sum-stepper__value mono">{numSentences}</output>
            <button type="button" className="sum-stepper__btn" onClick={() => setNumSentences((n) => Math.min(50, n + 1))} aria-label="More sentences">+</button>
          </div>
          <span className="sum-stepper__bound mono">1–50 sentences</span>
        </div>

        <div className="sum-run">
          <Button variant="primary" size="lg" loading={loading} disabled={!text.trim()} onClick={handleAnalyze}>
            {loading ? 'Extracting…' : 'Extract summary'}
          </Button>
          <Button variant="secondary" loading={comparing} disabled={!text.trim()} onClick={handleCompare}>
            Compare all 3
          </Button>
        </div>
      </section>

      <section className="sum-results" aria-label="Summary results" aria-live="polite">
        {loading || comparing ? (
          <div className="card card--pad"><SkeletonBlock lines={5} /></div>
        ) : null}

        {!loading && !comparing && !result && !compareResults && !error ? (
          <EmptyState
            icon="⌗"
            title="No summary yet"
            description="Pick an algorithm, choose how many sentences to keep, then extract. Or run all three side by side and compare."
          />
        ) : null}

        {!loading && !comparing && error && !result && !compareResults ? (
          <div className="callout-error" role="alert">{error}</div>
        ) : null}

        {!loading && !comparing && result ? (
          <SummaryResult
            result={result}
            sourceSentenceCount={stats.sentences}
            onListen={handleListen}
            audioLoading={audioLoading}
            audioError={audioError}
            onCopy={handleCopy}
            copied={copied}
            navigateActions={navigateActions}
          />
        ) : null}

        {!loading && !comparing && compareResults ? (
          <div className="sum-compare">
            {compareResults.map(({ method: m, result: r, error: e }) => (
              <div key={m.id} className="card sum-compare__col">
                {r ? (
                  <>
                    <div className="sum-compare__head">
                      <StatusBadge tone="accent">{r.method}</StatusBadge>
                      <span className="mono sum-compare__count">{r.num_sentences_selected} sent.</span>
                    </div>
                    <p className="sum-compare__text">{r.summary || '—'}</p>
                    <div className="sum-compare__foot">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          setResult(r);
                          setCompareResults(null);
                          setLastSummary(r);
                        }}
                      >
                        Focus
                      </Button>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="sum-compare__head"><StatusBadge tone="danger">{m.name}</StatusBadge></div>
                    <p className="sum-compare__text sum-compare__text--error">{e || 'Method failed.'}</p>
                  </>
                )}
              </div>
            ))}
          </div>
        ) : null}
      </section>

      <audio ref={audioRef} style={{ display: 'none' }} />
    </div>
  );
}
