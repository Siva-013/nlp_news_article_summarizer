import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { extractVocabulary, detectNewWords } from '../api/vocabulary.js';
import { Button } from '../components/ui/Button.jsx';
import { Field } from '../components/ui/Field.jsx';
import { EmptyState } from '../components/ui/EmptyState.jsx';
import { SkeletonRows } from '../components/ui/Skeleton.jsx';
import { StatusBadge, PosBadge } from '../components/ui/StatusBadge.jsx';
import { useToast } from '../components/ui/ToastContext.jsx';
import { useAppState } from '../context/AppStateContext.jsx';
import './Vocabulary.css';

export function Vocabulary() {
  const navigate = useNavigate();
  const toast = useToast();
  const {
    lastArticleText,
    lastSummary,
    setLastSummary,
    lastVocabResult,
    setLastVocabResult,
    lastDetectResult,
    setLastDetectResult,
    setPendingNote,
  } = useAppState();

  const [text, setText] = useState(lastArticleText || lastSummary?.summary || '');
  const [query, setQuery] = useState('');
  const [knownChips, setKnownChips] = useState([]);
  const [knownDraft, setKnownDraft] = useState('');
  const [maxItems, setMaxItems] = useState(30);
  const [includeEntities, setIncludeEntities] = useState(true);
  const [loading, setLoading] = useState(false);
  const [detecting, setDetecting] = useState(false);
  const [error, setError] = useState('');

  const items = lastVocabResult?.items || [];
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter((raw) =>
      [raw.term, raw.normalized_form, raw.lemma, raw.pos, raw.entity_label]
        .filter(Boolean)
        .join(' ')
        .toLowerCase()
        .includes(q),
    );
  }, [items, query]);

  function addKnownChip(rawValue) {
    const source = rawValue != null ? String(rawValue) : knownDraft;
    const parts = source.split(/[\n,]/).map((w) => w.trim()).filter(Boolean);
    if (parts.length === 0) return;
    setKnownChips((prev) => {
      const seen = new Set(prev.map((c) => c.toLowerCase()));
      return [...prev, ...parts.filter((p) => !seen.has(p.toLowerCase()))];
    });
    setKnownDraft('');
  }

  function removeKnownChip(chip) {
    setKnownChips((prev) => prev.filter((c) => c !== chip));
  }

  async function handleExtract() {
    const trimmed = text.trim();
    if (!trimmed) {
      setError('Provide text to extract vocabulary from.');
      return;
    }
    setError('');
    setLoading(true);
    setLastDetectResult(null);
    try {
      const res = await extractVocabulary({ text: trimmed, max_items: maxItems, include_entities: includeEntities });
      setLastVocabResult(res);
      toast(`Extracted ${res.returned_item_count} terms`, 'success');
    } catch (err) {
      setError(err.detail || err.message || 'Vocabulary extraction failed.');
      toast('Vocabulary extraction failed', 'error');
    } finally {
      setLoading(false);
    }
  }

  async function handleDetect() {
    const trimmed = text.trim();
    if (!trimmed && !lastVocabResult) {
      setError('Extract vocabulary or provide source text first.');
      return;
    }
    setError('');
    setDetecting(true);
    try {
      const res = await detectNewWords({
        text: trimmed || undefined,
        extraction_record: !trimmed && lastVocabResult ? lastVocabResult : undefined,
        known_vocabulary: knownChips,
        max_items: maxItems,
        include_entities: includeEntities,
      });
      setLastDetectResult(res);
      toast(`${res.new_word_count} new / ${res.known_word_count} known`, 'info');
    } catch (err) {
      setError(err.detail || err.message || 'New-word detection failed.');
      toast('New-word detection failed', 'error');
    } finally {
      setDetecting(false);
    }
  }

  function statusFor(item) {
    if (!lastDetectResult) return null;
    const matches = (list) =>
      list?.some((w) =>
        w.term === item.term && w.lemma === item.lemma,
      );
    if (matches(lastDetectResult.new_words)) return 'new';
    if (matches(lastDetectResult.known_words)) return 'known';
    return null;
  }

  function sendTermToNotes(item) {
    setPendingNote({
      term: item.term,
      content: `${item.term} — ${item.pos || 'word'}${item.entity_label ? ` (${item.entity_label})` : ''}. ${item.selection_reason || ''}`.trim(),
      example_sentence: '',
    });
    navigate('/notes');
  }

  const detectStats = lastDetectResult
    ? {
        total: lastDetectResult.total_vocabulary_count || 0,
        known: lastDetectResult.known_word_count || 0,
        fresh: lastDetectResult.new_word_count || 0,
      }
    : null;

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <p className="eyebrow">Vocabulary extraction</p>
          <h1 className="page-head__title">Build your <span className="text-gradient">word list</span></h1>
          <p className="page-head__sub">
            Rule-based extraction of content words and named entities, scored by frequency,
            sentence coverage, and entity weight.
          </p>
        </div>
        {lastVocabResult ? (
          <div className="page-head__aside mono">
            {lastVocabResult.returned_item_count} returned · {lastVocabResult.total_candidate_items} candidates
          </div>
        ) : null}
      </header>

      <div className="vocab-grid">
        <section className="card" aria-label="Vocabulary source">
          <div className="card__head">
            <span className="card__title">Source text</span>
            <div className="vocab-source-actions">
              <Button variant="ghost" size="sm" disabled={!lastArticleText} onClick={() => setText(lastArticleText)}>
                Use article
              </Button>
              <Button variant="ghost" size="sm" disabled={!lastSummary?.summary} onClick={() => setText(lastSummary.summary)}>
                Use summary
              </Button>
            </div>
          </div>
          <div className="vocab-source">
            <Field
              id="vocab-source"
              multiline
              rows={8}
              value={text}
              onChange={(e) => { setText(e.target.value); setError(''); }}
              onClear={() => setText('')}
              maxLength={50000}
              error={error}
            />
            <div className="vocab-params">
              <label className="field__label" htmlFor="max-items">Max items</label>
              <input
                id="max-items"
                className="vocab-params__number mono"
                type="number"
                min={1}
                max={200}
                value={maxItems}
                onChange={(e) => setMaxItems(Math.min(200, Math.max(1, Number(e.target.value) || 1)))}
              />
              <label className="vocab-params__check">
                <input type="checkbox" checked={includeEntities} onChange={(e) => setIncludeEntities(e.target.checked)} />
                <span>Include named entities</span>
              </label>
              <div className="vocab-params__run">
                <Button variant="primary" loading={loading} disabled={!text.trim()} onClick={handleExtract}>
                  {loading ? 'Extracting…' : 'Extract vocabulary'}
                </Button>
              </div>
            </div>
          </div>
        </section>

        <aside className="card" aria-label="Known vocabulary and new-word detection">
          <div className="card__head">
            <span className="card__title">New-word detection</span>
            <span className="card__hint">optional</span>
          </div>
          <div className="vocab-known">
            <p className="vocab-known__blurb">
              Add words you already know, then detect which extracted terms are new to you.
            </p>
            <div className="vocab-chips" aria-label="Known words">
              {knownChips.length === 0 ? (
                <span className="vocab-chips__empty">No known words added yet.</span>
              ) : (
                knownChips.map((chip) => (
                  <button key={chip} type="button" className="vocab-chip" onClick={() => removeKnownChip(chip)} title="Remove">
                    {chip}<span aria-hidden="true">×</span>
                  </button>
                ))
              )}
            </div>
            <input
              className="vocab-chip-input"
              value={knownDraft}
              onChange={(e) => setKnownDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ',') {
                  e.preventDefault();
                  addKnownChip(e.currentTarget.value);
                }
              }}
              onBlur={(e) => addKnownChip(e.currentTarget.value)}
              placeholder="Type a word, press Enter…"
              aria-label="Add known word"
            />
            <Button variant="secondary" loading={detecting} onClick={handleDetect} disabled={!text.trim() && !lastVocabResult}>
              {detecting ? 'Detecting…' : 'Detect new words'}
            </Button>

            {detectStats ? (
              <div className="vocab-detect-stats">
                <div className="vocab-detect-stats__row">
                  <span className="mono">total</span><b className="mono">{detectStats.total}</b>
                </div>
                <div className="meter vocab-detect-stats__meter">
                  <div className="meter__fill" style={{ width: `${detectStats.total ? (detectStats.known / detectStats.total) * 100 : 0}%` }} />
                </div>
                <div className="vocab-detect-stats__legend">
                  <span className="vocab-dot vocab-dot--known" /> known <b className="mono">{detectStats.known}</b>
                  <span className="vocab-dot vocab-dot--new" /> new <b className="mono">{detectStats.fresh}</b>
                </div>
              </div>
            ) : null}
          </div>
        </aside>
      </div>

      <section className="card" aria-label="Extracted vocabulary" aria-live="polite">
        <div className="card__head">
          <span className="card__title">Extracted terms</span>
          <input
            className="vocab-filter"
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter terms…"
            aria-label="Filter extracted vocabulary"
          />
        </div>

        <div className="vocab-list-wrap">
          {loading ? <div className="vocab-list-pad"><SkeletonRows rows={6} /></div> : null}

          {!loading && !lastVocabResult ? (
            <EmptyState
              icon="Aa"
              title="No extraction yet"
              description="Paste article text on the left and run the extractor. Terms appear here with scores and part-of-speech tags."
            />
          ) : null}

          {!loading && lastVocabResult && filtered.length === 0 ? (
            <EmptyState icon="◌" title="No matching terms" description="The current filter does not match any returned vocabulary items." />
          ) : null}

          {!loading && filtered.length > 0 ? (
            <ul className="vocab-list">
              {filtered.map((item, index) => {
                const flag = statusFor(item);
                const scorePct = Math.min(100, (Number(item.importance_score) || 0) * 100);
                return (
                  <li key={`${item.term}-${item.lemma}-${index}`} className={`vocab-row ${flag ? `vocab-row--${flag}` : ''}`}>
                    <div className="vocab-row__main">
                      <span className="vocab-row__term">{item.term}</span>
                      <PosBadge pos={item.pos} />
                      {item.entity_label ? <StatusBadge tone="info">{item.entity_label}</StatusBadge> : null}
                      {flag === 'new' ? <StatusBadge tone="warning">new</StatusBadge> : null}
                      {flag === 'known' ? <StatusBadge tone="success">known</StatusBadge> : null}
                    </div>
                    <div className="vocab-row__meta mono">
                      <span title="Occurrences">freq {item.frequency ?? '—'}</span>
                      <span title="Sentences containing the term">in {item.sentence_frequency ?? '—'} sent.</span>
                      {item.lemma && item.lemma !== item.term.toLowerCase() ? <span>lemma {item.lemma}</span> : null}
                    </div>
                    <div className="vocab-row__score" title={`Importance ${Number(item.importance_score || 0).toFixed(4)}`}>
                      <div className="meter"><div className="meter__fill" style={{ width: `${Math.max(3, scorePct)}%` }} /></div>
                      <span className="mono">{Number(item.importance_score || 0).toFixed(3)}</span>
                    </div>
                    <Button variant="ghost" size="sm" className="vocab-row__note" onClick={() => sendTermToNotes(item)} title="Create a note for this term">
                      + note
                    </Button>
                  </li>
                );
              })}
            </ul>
          ) : null}
        </div>
      </section>
    </div>
  );
}
