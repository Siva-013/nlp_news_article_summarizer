import { useCallback, useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { deleteHistory, listHistory, searchHistory } from '../api/history.js';
import { Button } from '../components/ui/Button.jsx';
import { ConfirmDialog } from '../components/ui/ConfirmDialog.jsx';
import { EmptyState } from '../components/ui/EmptyState.jsx';
import { SkeletonRows } from '../components/ui/Skeleton.jsx';
import { StatusBadge } from '../components/ui/StatusBadge.jsx';
import { useToast } from '../components/ui/ToastContext.jsx';
import { useAppState } from '../context/AppStateContext.jsx';
import './History.css';

const PAGE_SIZE = 20;
const METHOD_FILTERS = ['frequency', 'tfidf', 'textrank'];

function formatDate(iso) {
  try {
    return new Date(iso).toLocaleString(undefined, {
      year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    });
  } catch {
    return iso || '';
  }
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

  useEffect(() => {
    const q = searchParams.get('q');
    if (q) {
      setQuery(q);
      load(0, q, '');
    } else {
      load(0, '', '');
    }
    // Run once on mount; ?q= changes come with a fresh mount via the strip.
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
    } catch (err) {
      toast(err.detail || err.message || 'Delete failed', 'error');
    }
  }

  const page = Math.floor(offset / PAGE_SIZE) + 1;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const searching = Boolean(query.trim());

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <p className="eyebrow">Article archive</p>
          <h1 className="page-head__title">Summary <span className="text-gradient">history</span></h1>
          <p className="page-head__sub">
            Every summarization run is saved to SQLite. Open one to reload it in the workspace,
            send it to the listening studio, or remove it.
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
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search title, summary, or article text…"
            aria-label="Search article history"
          />
          <Button type="submit" variant="secondary" size="sm">Search</Button>
          {searching ? (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => { setQuery(''); load(0, '', methodFilter); }}
            >
              Clear
            </Button>
          ) : null}
        </form>

        <div className="history-filters" role="group" aria-label="Filter by method">
          {METHOD_FILTERS.map((m) => (
            <button
              key={m}
              type="button"
              className={`history-filter ${methodFilter === m ? 'history-filter--active' : ''} ${searching ? 'history-filter--disabled' : ''}`}
              onClick={() => applyMethod(m)}
              disabled={searching}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {loading ? <div className="card card--pad"><SkeletonRows rows={5} /></div> : null}
      {error ? <div className="callout-error" role="alert">{error}</div> : null}

      {!loading && items.length === 0 ? (
        <EmptyState
          icon="⛭"
          title={searching ? 'No matching entries' : 'No archived summaries yet'}
          description={
            searching
              ? 'Try a different search term, or clear the search to browse everything.'
              : 'Extract a summary in the workspace — successful runs are archived here automatically.'
          }
          action={<Button variant="primary" onClick={() => navigate('/')}>Go to workspace</Button>}
        />
      ) : null}

      {!loading && items.length > 0 ? (
        <>
          <ol className="history-list">
            {items.map((entry) => (
              <li key={entry.id} className="card history-card">
                <div className="history-card__head">
                  <div className="history-card__id mono">#{String(entry.id).padStart(3, '0')}</div>
                  <div className="history-card__titles">
                    <strong className="history-card__title">{entry.title || 'Untitled article'}</strong>
                    <span className="history-card__meta">
                      <StatusBadge tone="accent">{entry.method}</StatusBadge>
                      <span className="mono history-card__date">{formatDate(entry.processing_timestamp)}</span>
                      {entry.extracted_vocabulary?.length ? (
                        <span className="mono history-card__vocab">{entry.extracted_vocabulary.length} terms</span>
                      ) : null}
                    </span>
                  </div>
                </div>

                <p className="history-card__summary">{entry.summary}</p>

                {entry.article_text ? (
                  <button
                    type="button"
                    className="history-card__expand"
                    onClick={() => setExpanded(expanded === entry.id ? null : entry.id)}
                    aria-expanded={expanded === entry.id}
                  >
                    {expanded === entry.id ? 'Hide article text' : 'Show article text'}
                  </button>
                ) : null}
                {expanded === entry.id ? (
                  <p className="history-card__article">{entry.article_text}</p>
                ) : null}

                <div className="history-card__actions">
                  <Button variant="secondary" size="sm" onClick={() => openEntry(entry)}>Open</Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => { setTtsPendingText(entry.summary); navigate('/listen'); }}
                  >
                    Listen ▶
                  </Button>
                  <Button variant="danger" size="sm" onClick={() => setPendingDelete(entry)}>Delete</Button>
                </div>
              </li>
            ))}
          </ol>

          <div className="history-pager">
            <Button
              variant="ghost"
              size="sm"
              disabled={offset <= 0}
              onClick={() => load(Math.max(0, offset - PAGE_SIZE), query, methodFilter)}
            >
              ← Prev
            </Button>
            <span className="mono history-pager__info">
              page {page} / {pages} · {total} total
            </span>
            <Button
              variant="ghost"
              size="sm"
              disabled={offset + PAGE_SIZE >= total}
              onClick={() => load(offset + PAGE_SIZE, query, methodFilter)}
            >
              Next →
            </Button>
          </div>
        </>
      ) : null}

      <ConfirmDialog
        open={Boolean(pendingDelete)}
        title="Delete this history entry?"
        description="The archive record is removed. Linked notes keep their text but lose the article association."
        onCancel={() => setPendingDelete(null)}
        onConfirm={confirmDelete}
      />
    </div>
  );
}
