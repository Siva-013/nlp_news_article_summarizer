import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { summarize } from '../api/summarize.js';
import { createHistory, listHistory } from '../api/history.js';
import { synthesizeSpeech, getAudioUrl } from '../api/speech.js';
import { Button } from '../components/ui/Button.jsx';
import { Field } from '../components/ui/Field.jsx';
import { SkeletonBlock } from '../components/ui/Skeleton.jsx';
import { EmptyState } from '../components/ui/EmptyState.jsx';
import { StatusBadge } from '../components/ui/StatusBadge.jsx';
import { useToast } from '../components/ui/ToastContext.jsx';
import { useAppState } from '../context/AppStateContext.jsx';
import { useHealth } from '../context/HealthContext.jsx';
import { MarkerUnderline, CircleScribble, Starburst, Squiggle, DoodleArrow } from '../components/doodles.jsx';
import './Summarizer.css';

const METHODS = [
  { id: 'frequency', name: 'Frequency', blurb: 'Luhn-style normalized word frequency scoring.' },
  { id: 'tfidf', name: 'TF–IDF', blurb: 'Term rarity weighted across the document.' },
  { id: 'textrank', name: 'TextRank', blurb: 'Sentence graph centrality via PageRank.' },
];

const SAMPLE_ARTICLE = `Scientists have developed a new type of battery that can store significantly more energy than current lithium-ion batteries. The research team at MIT published their findings in the journal Nature Energy, detailing how the solid-state battery uses a special ceramic electrolyte instead of liquid. The new design eliminates fire risk while doubling energy density. Commercial applications could emerge within five years, according to project lead Dr. Sarah Chen. Auto manufacturers including Toyota and Ford have already expressed interest in licensing the technology for electric vehicles.`;

const HERO_CHIPS = [
  '3 algorithms',
  '100% offline',
  'verbatim extracts',
  'auto archive',
];

function countSentences(text) {
  if (!text.trim()) return 0;
  return text.split(/(?<=[.!?])\s+/).filter((s) => s.trim().length > 0).length;
}

function formatDate(iso) {
  try {
    return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch {
    return '';
  }
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
    <details className="fold">
      <summary>Top scoring words <span className="fold__hint">({method === 'tfidf' ? 'TF–IDF weight' : 'normalized frequency'})</span></summary>
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
          {compression != null ? <StatusBadge tone="neutral">{compression}% of source</StatusBadge> : null}
        </div>
      </div>

      <div className="sum-result__body">
        <p className="sum-result__text">{result.summary || '—'}</p>

        {result.selected_sentences?.length > 0 ? (
          <details className="fold" open>
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

function StatusCard() {
  const { status, database } = useHealth();
  const api = status === 'online';
  const rows = [
    { name: 'API SERVER', detail: api ? 'online' : 'offline', ok: api },
    { name: 'DATABASE', detail: database ? 'connected' : 'offline', ok: Boolean(database) },
    { name: 'TTS ENGINE', detail: api ? 'ready' : 'offline', ok: api },
    { name: 'ALGORITHMS', detail: '3 loaded', ok: true },
  ];
  return (
    <div className="card statuscard">
      <div className="card__head"><span className="card__title">System status</span></div>
      <ul className="statuscard__list">
        {rows.map((r) => (
          <li key={r.name} className="statuscard__row">
            <span className="statuscard__name">{r.name}</span>
            <span className="statuscard__dots" aria-hidden="true" />
            <span className={`statuscard__val ${r.ok ? 'statuscard__val--ok' : 'statuscard__val--bad'}`}>
              {r.ok ? '●' : '○'} {r.detail}
            </span>
          </li>
        ))}
      </ul>
      <p className="statuscard__note mono">polled every 20s</p>
    </div>
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
  const [recent, setRecent] = useState([]);
  const audioRef = useRef(null);
  const navigate = useNavigate();
  const toast = useToast();
  const {
    setLastArticleText,
    setLastSummary,
    setTtsPendingText,
    setPendingNote,
    setLastHistoryId,
    lastVocabResult,
  } = useAppState();

  const stats = useMemo(() => ({
    words: text.trim() ? text.trim().split(/\s+/).length : 0,
    chars: text.length,
    sentences: countSentences(text),
  }), [text]);

  useEffect(() => {
    listHistory({ limit: 3, offset: 0 })
      .then((res) => setRecent(res.items || []))
      .catch(() => setRecent([]));
  }, [result]);

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
      {/* ---------- Hero ---------- */}
      <header className="hero">
        <div className="hero__copy">
          <p className="eyebrow">Field notes · 01 / extractive summarization</p>
          <h1 className="hero__title">
            Turn long articles into{' '}
            <span className="hero__hl">
              clear insights
              <MarkerUnderline className="hero__underline" />
            </span>
            .
          </h1>
          <p className="hero__lede">
            LexiNLP lifts the most important sentences — verbatim, explainably, and entirely
            on this machine. No cloud, no black boxes.
          </p>
          <ul className="hero__chips" aria-label="Highlights">
            {HERO_CHIPS.map((c) => (
              <li key={c} className="hero__chip">{c}</li>
            ))}
          </ul>
        </div>

        <div className="hero__collage" aria-hidden="true">
          <Starburst className="hero__star" size={30} />
          <div className="collage-card collage-card--summary">
            <span className="collage-card__tag mono">summary</span>
            <p className="collage-card__line">Scientists developed a new battery…</p>
            <p className="collage-card__line collage-card__line--dim">The design eliminates fire risk…</p>
            <span className="collage-tape collage-tape--a" />
          </div>
          <div className="collage-card collage-card--vocab">
            <span className="collage-card__tag mono">vocabulary</span>
            <div className="collage-vocab">
              <span>battery</span>
              <span>energy</span>
              <span>electrolyte</span>
            </div>
            <CircleScribble className="collage-scribble" />
          </div>
          <div className="collage-card collage-card--wave">
            <span className="collage-card__tag mono">audio</span>
            <Squiggle className="collage-wave" />
          </div>
          <DoodleArrow className="hero__arrow" />
        </div>
      </header>

      {/* ---------- Workspace ---------- */}
      <div className="sum-grid">
        <div className="sum-main">
          <section className="card" aria-label="Article input">
            <div className="card__head">
              <span className="section-head"><span className="section-mark">01</span> <span className="card__title">Start with an article</span></span>
              <div className="stat-row">
                <span className="stat-chip">words <b>{stats.words}</b></span>
                <span className="stat-chip">sentences <b>{stats.sentences}</b></span>
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

          <section className="card" aria-label="Algorithm and length settings">
            <div className="card__head">
              <span className="section-head"><span className="section-mark">02</span> <span className="card__title">Choose a method &amp; extract</span></span>
              <span className="mono card__hint">ctrl+enter to run</span>
            </div>
            <div className="sum-controls">
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

              <div className="sum-run">
                <div className="sum-length">
                  <label className="field__label" htmlFor="num-sentences">Length</label>
                  <div className="sum-stepper">
                    <button type="button" className="sum-stepper__btn" onClick={() => setNumSentences((n) => Math.max(1, n - 1))} aria-label="Fewer sentences">−</button>
                    <output id="num-sentences" className="sum-stepper__value mono">{numSentences}</output>
                    <button type="button" className="sum-stepper__btn" onClick={() => setNumSentences((n) => Math.min(50, n + 1))} aria-label="More sentences">+</button>
                  </div>
                </div>
                <Button variant="primary" size="lg" loading={loading} disabled={!text.trim()} onClick={handleAnalyze}>
                  {loading ? 'Extracting…' : 'Generate summary'}
                </Button>
                <Button variant="ghost" loading={comparing} disabled={!text.trim()} onClick={handleCompare}>
                  Compare all 3
                </Button>
              </div>
            </div>
          </section>

          <section className="sum-results" aria-label="Summary results" aria-live="polite">
            {loading || comparing ? (
              <div className="card card--pad"><SkeletonBlock lines={5} /></div>
            ) : null}

            {!loading && !comparing && !result && !compareResults && !error ? (
              <EmptyState
                icon="¶"
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
                  <div key={m.id} className={`card sum-compare__col ${r ? '' : 'sum-compare__col--bad'}`}>
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
        </div>

        {/* ---------- Right rail ---------- */}
        <aside className="sum-rail">
          <div className="card quickcard">
            <div className="card__head"><span className="card__title">Quick actions</span></div>
            <ul className="quickcard__list">
              <li><Link to="/vocabulary">Extract vocabulary <span aria-hidden="true">→</span></Link></li>
              <li><Link to="/listen">Listen to a summary <span aria-hidden="true">→</span></Link></li>
              <li><Link to="/notes">Open my notes <span aria-hidden="true">→</span></Link></li>
              <li><Link to="/evaluate">Score with ROUGE <span aria-hidden="true">→</span></Link></li>
            </ul>
          </div>

          <StatusCard />

          <div className="quotecard">
            <span className="quotecard__mark" aria-hidden="true">“</span>
            <p className="quotecard__text">
              Nothing is rewritten — every sentence is lifted verbatim from the source article.
            </p>
            <Squiggle className="quotecard__squiggle" />
          </div>
        </aside>
      </div>

      {/* ---------- Bottom band ---------- */}
      <section className="sum-band" aria-label="Overview">
        <div className="card bandcard">
          <div className="card__head"><span className="card__title">Supported algorithms</span></div>
          <ul className="bandcard__algos">
            {METHODS.map((m) => (
              <li key={m.id}>
                <span className="bandcard__algo-name">{m.name}</span>
                <span className="bandcard__algo-blurb">{m.blurb}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="card bandcard">
          <div className="card__head">
            <span className="card__title">Recent activity</span>
            <Link to="/history" className="mono bandcard__link">view all →</Link>
          </div>
          {recent.length === 0 ? (
            <p className="bandcard__empty">Your archived summaries will appear here.</p>
          ) : (
            <ul className="bandcard__recent">
              {recent.map((r) => (
                <li key={r.id}>
                  <span className="bandcard__recent-title">{r.title || 'Untitled article'}</span>
                  <span className="mono bandcard__recent-meta">{r.method} · {formatDate(r.processing_timestamp)}</span>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="card bandcard">
          <div className="card__head">
            <span className="card__title">Vocabulary preview</span>
            <Link to="/vocabulary" className="mono bandcard__link">open →</Link>
          </div>
          {lastVocabResult?.items?.length ? (
            <ul className="bandcard__vocab">
              {lastVocabResult.items.slice(0, 5).map((item, i) => (
                <li key={`${item.term}-${i}`}>{item.term}</li>
              ))}
            </ul>
          ) : (
            <p className="bandcard__empty">Extract vocabulary from any article to start your word list.</p>
          )}
        </div>
      </section>

      <audio ref={audioRef} style={{ display: 'none' }} />
    </div>
  );
}
