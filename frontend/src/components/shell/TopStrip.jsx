import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './TopStrip.css';

/**
 * Thin strip above the content: breadcrumb tabs (per-section context),
 * a global search that jumps into History search, and the menu button
 * shown on narrow screens.
 */
export function TopStrip({ section, crumbs = [], onMenuOpen }) {
  const navigate = useNavigate();
  const [q, setQ] = useState('');

  function submitSearch(e) {
    e.preventDefault();
    const term = q.trim();
    navigate(term ? `/history?q=${encodeURIComponent(term)}` : '/history');
  }

  return (
    <div className="strip">
      <button
        type="button"
        className="strip__menu"
        onClick={onMenuOpen}
        aria-label="Open navigation"
      >
        ☰
      </button>
      <div className="strip__crumbs" aria-label="Breadcrumb">
        <span className="strip__crumb-strip">{section}</span>
        {crumbs.map((c, i) => (
          <span className="strip__crumb" key={c}>
            {i > 0 ? <span className="strip__crumb-sep" aria-hidden="true">/</span> : null}
            {c}
          </span>
        ))}
      </div>
      <form className="strip__search" onSubmit={submitSearch} role="search">
        <span className="strip__search-icon" aria-hidden="true">⌕</span>
        <input
          type="search"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search the archive…"
          aria-label="Search article history"
        />
      </form>
    </div>
  );
}
