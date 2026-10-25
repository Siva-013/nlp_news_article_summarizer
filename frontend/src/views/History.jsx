import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { deleteHistory, listHistory, searchHistory } from '../api/history.js';
import { Button } from '../components/ui/Button.jsx';
import { ConfirmDialog } from '../components/ui/ConfirmDialog.jsx';
import { EmptyState } from '../components/ui/EmptyState.jsx';
import { SkeletonRows } from '../components/ui/Skeleton.jsx';
import { useToast } from '../components/ui/ToastContext.jsx';
import { useAppState } from '../context/AppStateContext.jsx';
import { Squiggle } from '../components/doodles.jsx';
import './History.css';

const PAGE_SIZE = 20;

const METHOD_FILTERS = [
  { id: 'frequency', label: 'Frequency', tone: 'badge--info' },
  { id: 'tfidf', label: 'TF–IDF', tone: 'badge--warning' },
  { id: 'textrank', label: 'TextRank', tone: 'badge--success' },
];

function wordCount(text) {
  return text?.trim() ? text.trim().split(/\s+/).length : 0;
}

function formatTime(iso) {
  try {
    return new Date(iso).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
  } catch {
    return '';
  }
}

function dayLabel(iso) {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      weekday: 'long', month: 'long', day: 'numeric',
    });
  } catch {
    return iso || '';
  }
}

const METHOD_TONES = Object.fromEntries(METHOD_FILTERS.map((m) => [m.id, m.tone]));

/** Wraps case-insensitive matches of the query in <mark> highlights. */
function Highlight({ text, query }) {
  const source = String(text ?? '');
  const needle = query?.trim();
  if (!needle || !source) return source;
  const lower = source.toLowerCase();
  const target = needle.toLowerCase();
  const out = [];
  let from = 0;
  let at;
  let key = 0;
  while ((at = lower.indexOf(target, from)) !== -1) {
    if (at > from) out.push(source.slice(from, at));
    out.push(<mark key={key++} className="hl">{source.slice(at, at + needle.length)}</mark>);
    from = at + needle.length;
  }
  out.push(source.slice(from));
  return out;
}

function Stamp({ method }) {
  return (
    <span className={`stamp ${METHOD_TONES[method] || 'badge--neutral'}`}>
      {method}
    </span>
  );
}

export function History() {
  const navigate = useNavigate();
  const toast = useToast();
  const [searchParams] = useSearchParams();
  const { setLastArticleText, setLastSummary, setLastHistoryId, setTtsPendingText } = useAppState();
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [query, setQuery] = useState('');
  const [methodFilter, setMethodFilter] = useState('');
  const [counts, setCounts] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [pendingDelete, setPendingDelete] = useState(null);
  const [expanded, setExpanded] = useState(null);

  const load = useCallback(async (nextOffset, search, method) => {
    setLoading(true);
    setError('');
    try {
      const res = search?.trim()
        ? await searchHistory({ q: search.trim(), limit: PAGE_SIZE, offset: nextOffset })
        : await listHistory({ limit: PAGE_SIZE, offset: nextOffset, method: method || undefined });
      setItems(res.items || []);
      setTotal(res.total ?? (res.items || []).length);
      setOffset(res.offset ?? nextOffset);
    } catch (err) {
      setError(err.detail || err.message || 'Could not load history.');
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadCounts = useCallback(async () => {
    try {
      const res = await Promise.all(
        METHOD_FILTERS.map((m) => listHistory({ limit: 1, offset: 0, method: m.id })),
      );
      setCounts(Object.fromEntries(METHOD_FILTERS.map((m, i) => [m.id, res[i].total ?? 0])));
    } catch {
      setCounts(null);
    }
  }, []);

  useEffect(() => {
    const q = searchParams.get('q');
    if (q) {
      setQuery(q);
      load(0, q, '');
    } else {
      load(0, '', '');
    }
    loadCounts();
    // Run once on mount; ?q= arrives with a fresh mount via the strip.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function applyMethod(method) {
    const next = methodFilter === method ? '' : method;
    setMethodFilter(next);
    setQuery('');
    load(0, '', next);
  }

  function openEntry(entry) {
    setLastHistoryId(entry.id);
    setLastArticleText(entry.article_text || '');
    setLastSummary({
      article_id: entry.source_article_id || String(entry.id),
      method: entry.method,
      summary: entry.summary,
      num_sentences_requested: entry.metadata?.num_sentences_requested,
      num_sentences_selected: entry.metadata?.num_sentences_selected,
      selected_sentences: [],
    });
    navigate('/');
  }

  async function confirmDelete() {
    if (!pendingDelete) return;
    try {
      await deleteHistory(pendingDelete.id);
      toast('History entry deleted', 'success');
      setPendingDelete(null);
      await load(offset, query, methodFilter);
      await loadCounts();
    } catch (err) {
      toast(err.detail || err.message || 'Delete failed', 'error');
    }
  }

  // Group the page's entries under day headers, like a journal.
  const groups = useMemo(() => {
    const map = new Map();
    for (const entry of items) {
      const key = dayLabel(entry.processing_timestamp);
      let arr = map.get(key);
      if (!arr) {
        arr = [];
        map.set(key, arr);
      }
      arr.push(entry);
    }
    return [...map.entries()];
  }, [items]);

  const page = Math.floor(offset / PAGE_SIZE) + 1;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const searching = Boolean(query.trim());
  const rangeStart = total === 0 ? 0 : offset + 1;
  const rangeEnd = Math.min(offset + items.length, total);

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <p className="eyebrow">Article archive</p>
          <h1 className="page-head__title">Summary history</h1>
          <p className="page-head__sub">
            Every summarization run is pressed into this ledger. Open one to reload it in the
            workspace, send it to the listening studio, or tear it out.
          </p>
        </div>
        <div className="page-head__aside mono">{total} entries</div>
      </header>

      <div className="history-toolbar">
        <form
          className="history-search"
          onSubmit={(e) => {
            e.preventDefault();
            setMethodFilter('');
            load(0, query, '');
          }}
        >
          <span className="history-search__icon" aria-hidden="true">⌕</span>
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search the ledger — titles, summaries, article text…"
            aria-label="Search article history"
          />
          {searching ? (
            <Button type="button" variant="ghost" size="sm" onClick={() => { setQuery(''); load(0, '', methodFilter); }}>
              Clear ✕
            </Button>
          ) : null}
        </form>

        <div className="history-filters" role="group" aria-label="Filter by method">
          <button
            type="button"
            className={`history-filter ${!methodFilter && !searching ? 'history-filter--active' : ''} ${searching ? 'history-filter--disabled' : ''}`}
            onClick={() => { setMethodFilter(''); load(0, '', ''); }}
            disabled={searching}
          >
            All
          </button>
          {METHOD_FILTERS.map((m) => (
            <button
              key={m.id}
              type="button"
              className={`history-filter ${methodFilter === m.id ? 'history-filter--active' : ''} ${searching ? 'history-filter--disabled' : ''}`}
              onClick={() => applyMethod(m.id)}
              disabled={searching}
            >
              {m.label}
              {counts ? <span className="history-filter__count">{counts[m.id]}</span> : null}
            </button>
          ))}
        </div>
      </div>

      {loading ? <div className="card card--pad"><SkeletonRows rows={5} /></div> : null}
      {error ? <div className="callout-error" role="alert">{error}</div> : null}

      {!loading && items.length === 0 ? (
        <EmptyState
          icon="⌛"
          title={searching ? 'Nothing in the ledger matches' : methodFilter ? `No ${methodFilter} entries yet` : 'The ledger is empty'}
          description={
            searching
              ? `No entries contain “${query.trim()}”. Try a different word, or clear the search.`
              : 'Extract a summary in the workspace — successful runs are pressed into this ledger automatically.'
          }
          action={<Button variant="primary" onClick={() => navigate('/')}>Go to workspace</Button>}
        />
      ) : null}

      {!loading && items.length > 0 ? (
        <>
          <div className="card ledger">
            {groups.map(([day, entries]) => (
              <section key={day} className="ledger__day">
                <header className="ledger__day-head">
                  <span className="ledger__day-label mono">{day}</span>
                  <Squiggle className="ledger__day-squiggle" />
                </header>
                <ol className="ledger__list">
                  {entries.map((entry) => (
                    <li key={entry.id} className="ledger__row">
                      <div className="ledger__clipno mono" title={`History entry #${entry.id}`}>
                        #{String(entry.id).padStart(3, '0')}
                      </div>
                      <div className="ledger__body">
                        <div className="ledger__topline">
                          <h3 className="ledger__title">
                            <Highlight text={entry.title || 'Untitled article'} query={searching ? query : ''} />
                          </h3>
                          <Stamp method={entry.method} />
                          <span className="ledger__time mono">{formatTime(entry.processing_timestamp)}</span>
                        </div>

                        <p className="ledger__summary">
                          <Highlight text={entry.summary} query={searching ? query : ''} />
                        </p>

                        <p className="ledger__meta mono">
                          <span>{entry.metadata?.num_sentences_selected ?? '—'} sentences</span>
                          <span>{wordCount(entry.summary)}-word summary</span>
                          {entry.article_text ? <span>source {wordCount(entry.article_text)} words</span> : null}
                          {entry.extracted_vocabulary?.length ? <span>{entry.extracted_vocabulary.length} terms</span> : null}
                        </p>

                        {entry.article_text ? (
                          <button
                            type="button"
                            className="ledger__expand"
                            onClick={() => setExpanded(expanded === entry.id ? null : entry.id)}
                            aria-expanded={expanded === entry.id}
                          >
                            {expanded === entry.id ? 'Hide source text ▴' : 'Show source text ▾'}
                          </button>
                        ) : null}
                        {expanded === entry.id ? (
                          <p className="ledger__article">{entry.article_text}</p>
                        ) : null}

                        <div className="ledger__actions">
                          <Button variant="secondary" size="sm" onClick={() => openEntry(entry)}>
                            Open ↗
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => { setTtsPendingText(entry.summary); navigate('/listen'); }}
                          >
                            Listen ♪
                          </Button>
                          <Button variant="ghost" size="sm" onClick={() => setPendingDelete(entry)}>
                            Tear out
                          </Button>
                        </div>
                      </div>
                    </li>
                  ))}
                </ol>
              </section>
            ))}
          </div>

          <div className="history-pager">
            <Button
              variant="secondary"
              size="sm"
              disabled={offset <= 0}
              onClick={() => load(Math.max(0, offset - PAGE_SIZE), query, methodFilter)}
            >
              ← Newer
            </Button>
            <span className="mono history-pager__info">
              {rangeStart}–{rangeEnd} of {total} · page {page} / {pages}
            </span>
            <Button
              variant="secondary"
              size="sm"
              disabled={offset + PAGE_SIZE >= total}
              onClick={() => load(offset + PAGE_SIZE, query, methodFilter)}
            >
              Older →
            </Button>
          </div>
        </>
      ) : null}

      <ConfirmDialog
        open={Boolean(pendingDelete)}
        title="Tear out this page?"
        description="The archive record is removed. Linked notes keep their text but lose the article association."
        confirmLabel="Tear out"
        onCancel={() => setPendingDelete(null)}
        onConfirm={confirmDelete}
      />
    </div>
  );
}
