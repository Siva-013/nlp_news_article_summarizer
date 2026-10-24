import { useCallback, useEffect, useRef, useState } from 'react';
import { createNote, deleteNote, listNotes, searchNotes, updateNote } from '../api/notes.js';
import { Button } from '../components/ui/Button.jsx';
import { ConfirmDialog } from '../components/ui/ConfirmDialog.jsx';
import { SkeletonRows } from '../components/ui/Skeleton.jsx';
import { EmptyState } from '../components/ui/EmptyState.jsx';
import { StatusBadge } from '../components/ui/StatusBadge.jsx';
import { useToast } from '../components/ui/ToastContext.jsx';
import { useAppState } from '../context/AppStateContext.jsx';
import './Notes.css';

const BLANK = { id: null, term: '', content: '', example_sentence: '', article_history_id: null };

export function Notes() {
  const toast = useToast();
  const { pendingNote, setPendingNote, lastHistoryId } = useAppState();
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');
  const [selected, setSelected] = useState(BLANK);
  const [savedSnapshot, setSavedSnapshot] = useState(BLANK);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const debounceRef = useRef(null);

  const dirty =
    selected.term !== savedSnapshot.term ||
    selected.content !== savedSnapshot.content ||
    selected.example_sentence !== savedSnapshot.example_sentence;

  const load = useCallback(async (search) => {
    setLoading(true);
    setError('');
    try {
      const res = search?.trim()
        ? await searchNotes({ q: search.trim(), limit: 50, offset: 0 })
        : await listNotes({ limit: 50, offset: 0 });
      setItems(res.items || []);
      setTotal(res.total ?? (res.items || []).length);
    } catch (err) {
      setError(err.detail || err.message || 'Could not load notes.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load('');
  }, [load]);

  // Debounced typeahead search
  useEffect(() => {
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => load(query), 280);
    return () => clearTimeout(debounceRef.current);
  }, [query, load]);

  useEffect(() => {
    if (!pendingNote) return;
    const draft = {
      id: null,
      term: pendingNote.term || '',
      content: pendingNote.content || '',
      example_sentence: pendingNote.example_sentence || '',
      article_history_id: lastHistoryId,
    };
    setSelected(draft);
    setSavedSnapshot(BLANK);
    setPendingNote(null);
  }, [pendingNote, lastHistoryId, setPendingNote]);

  function startNew() {
    setSelected({ ...BLANK, article_history_id: lastHistoryId });
    setSavedSnapshot(BLANK);
  }

  function openNote(note) {
    const next = {
      id: note.id,
      term: note.term || '',
      content: note.content || '',
      example_sentence: note.example_sentence || '',
      article_history_id: note.article_history_id ?? null,
    };
    setSelected(next);
    setSavedSnapshot(next);
  }

  async function handleSave() {
    if (!selected.content.trim()) {
      toast('Note content cannot be empty', 'error');
      return;
    }
    setSaving(true);
    try {
      const payload = {
        content: selected.content.trim(),
        term: selected.term.trim() || undefined,
        example_sentence: selected.example_sentence.trim() || undefined,
        article_history_id: selected.article_history_id || undefined,
      };
      const saved = selected.id
        ? await updateNote(selected.id, {
            ...payload,
            clear_term: !selected.term.trim(),
            clear_example: !selected.example_sentence.trim(),
          })
        : await createNote(payload);
      const next = {
        id: saved.id,
        term: saved.term || '',
        content: saved.content || '',
        example_sentence: saved.example_sentence || '',
        article_history_id: saved.article_history_id ?? null,
      };
      setSelected(next);
      setSavedSnapshot(next);
      toast('Note saved', 'success');
      await load(query);
    } catch (err) {
      toast(err.detail || err.message || 'Save failed', 'error');
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete() {
    if (!selected.id) {
      startNew();
      setConfirmOpen(false);
      return;
    }
    try {
      await deleteNote(selected.id);
      toast('Note deleted', 'success');
      setConfirmOpen(false);
      startNew();
      await load(query);
    } catch (err) {
      toast(err.detail || err.message || 'Delete failed', 'error');
    }
  }

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <p className="eyebrow">Learner notebook</p>
          <h1 className="page-head__title">Vocabulary <span className="text-gradient">notes</span></h1>
          <p className="page-head__sub">
            Everything is stored in SQLite through the notes API. Search as you type — results
            cover terms, definitions, and example sentences.
          </p>
        </div>
        <div className="page-head__aside mono">{total} stored</div>
      </header>

      <div className="notes-layout">
        <aside className="card notes-index" aria-label="Notes list">
          <div className="card__head notes-index__head">
            <input
              type="search"
              className="notes-search"
              placeholder="Search notes…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              aria-label="Search notes"
            />
            <Button variant="secondary" size="sm" onClick={startNew}>+ New</Button>
          </div>

          <div className="notes-index__body">
            {loading ? <SkeletonRows rows={5} /> : null}
            {error ? <div className="callout-error notes-index__error">{error}</div> : null}
            {!loading && !error && items.length === 0 ? (
              <p className="notes-index__empty">
                {query ? `No notes match “${query}”.` : 'No saved notes yet — write one on the right.'}
              </p>
            ) : null}
            <ul className="notes-list">
              {!loading && items.map((note) => (
                <li key={note.id}>
                  <button
                    type="button"
                    className={`notes-item ${selected.id === note.id ? 'notes-item--active' : ''}`}
                    onClick={() => openNote(note)}
                  >
                    <span className="notes-item__term">{note.term || 'Untitled'}</span>
                    <span className="notes-item__preview">{note.content}</span>
                    <span className="notes-item__date mono">
                      {new Date(note.updated_at || note.created_at).toLocaleString()}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </aside>

        <section className="card notes-editor" aria-label="Note editor">
          <div className="card__head">
            <div className="notes-editor__status">
              {selected.id ? (
                <StatusBadge tone={dirty ? 'warning' : 'success'}>
                  {dirty ? 'Unsaved changes' : `Saved · #${selected.id}`}
                </StatusBadge>
              ) : (
                <StatusBadge tone="neutral">{dirty ? 'Unsaved draft' : 'New note'}</StatusBadge>
              )}
              {selected.article_history_id ? (
                <StatusBadge tone="info">article #{selected.article_history_id}</StatusBadge>
              ) : null}
            </div>
          </div>

          <div className="notes-editor__body">
            <label className="field__label" htmlFor="note-term">Term</label>
            <input
              id="note-term"
              className="notes-input notes-input--term"
              value={selected.term}
              onChange={(e) => setSelected((s) => ({ ...s, term: e.target.value }))}
              placeholder="e.g. satellite"
            />

            <label className="field__label" htmlFor="note-content">Definition / notes</label>
            <textarea
              id="note-content"
              className="notes-input notes-input--body"
              rows={10}
              value={selected.content}
              onChange={(e) => setSelected((s) => ({ ...s, content: e.target.value }))}
              placeholder="What does this word mean to you?"
            />

            <label className="field__label" htmlFor="note-example">Example sentence</label>
            <textarea
              id="note-example"
              className="notes-input notes-input--example"
              rows={3}
              value={selected.example_sentence}
              onChange={(e) => setSelected((s) => ({ ...s, example_sentence: e.target.value }))}
              placeholder="Where you met it, or a sentence of your own."
            />

            <div className="notes-editor__actions">
              <Button variant="primary" loading={saving} onClick={handleSave} disabled={!selected.content.trim()}>
                {selected.id ? 'Save changes' : 'Create note'}
              </Button>
              <Button variant="danger" onClick={() => setConfirmOpen(true)} disabled={!selected.id && !dirty}>
                Delete
              </Button>
            </div>
          </div>
        </section>
      </div>

      <ConfirmDialog
        open={confirmOpen}
        title="Delete this note?"
        description="This removes the note from the SQLite store. This cannot be undone."
        onCancel={() => setConfirmOpen(false)}
        onConfirm={handleDelete}
      />
    </div>
  );
}
