import { NavLink } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext.jsx';
import { useHealth } from '../../context/HealthContext.jsx';
import { Starburst, DoodleArrow } from '../doodles.jsx';
import './Sidebar.css';

export const NAV_ITEMS = [
  { to: '/', label: 'Summarize', icon: '¶', end: true },
  { to: '/vocabulary', label: 'Vocabulary', icon: 'Aa' },
  { to: '/notes', label: 'Notes', icon: '✎' },
  { to: '/history', label: 'History', icon: '⌛' },
  { to: '/listen', label: 'Listen', icon: '♪' },
  { to: '/evaluate', label: 'Evaluate', icon: '%' },
];

function StatusPanel() {
  const { status, database } = useHealth();
  const api = status === 'online' ? 'online' : status === 'degraded' ? 'degraded' : 'offline';
  const rows = [
    { name: 'API SERVER', value: api, ok: api === 'online' },
    { name: 'DATABASE', value: database ? 'connected' : 'offline', ok: Boolean(database) },
    { name: 'TTS ENGINE', value: api === 'online' ? 'ready' : 'offline', ok: api === 'online' },
  ];
  return (
    <div className="side-status">
      <p className="side-status__title">SYSTEM STATUS</p>
      {rows.map((r) => (
        <div className="side-status__row" key={r.name}>
          <span>{r.name}</span>
          <span className={`side-status__val ${r.ok ? 'side-status__val--ok' : 'side-status__val--bad'}`}>
            {r.ok ? '●' : '○'} {r.value}
          </span>
        </div>
      ))}
    </div>
  );
}

export function Sidebar({ open, onClose }) {
  const { user, logout } = useAuth();
  return (
    <>
      {open ? (
        <button type="button" className="side-backdrop" onClick={onClose} aria-label="Close navigation" />
      ) : null}
      <aside className={`sidebar ${open ? 'sidebar--open' : ''}`} aria-label="Primary">
        <div className="sidebar__inner">
          <div className="sidebar__brand">
            <Starburst size={26} />
            <span className="sidebar__wordmark">Lexi<span className="sidebar__wordmark-hl">NLP</span></span>
          </div>
          <p className="sidebar__tagline">
            News summarization &amp; vocabulary learning
          </p>

          <nav className="sidebar__nav">
            <p className="sidebar__nav-label">WORKSPACE</p>
            <ul className="sidebar__list">
              {NAV_ITEMS.map(({ to, label, icon, end }) => (
                <li key={to}>
                  <NavLink
                    to={to}
                    end={end}
                    className={({ isActive }) => `sidebar__link ${isActive ? 'sidebar__link--active' : ''}`}
                    onClick={open ? onClose : undefined}
                  >
                    <span className="sidebar__link-icon" aria-hidden="true">{icon}</span>
                    <span>{label}</span>
                    <span className="sidebar__link-dash" aria-hidden="true" />
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>

          <div className="sidebar__foot">
            <StatusPanel />
            <div className="sidebar__user">
              <span className="sidebar__avatar" aria-hidden="true">
                {(user?.username || '?').charAt(0).toUpperCase()}
              </span>
              <span className="sidebar__username">{user?.username}</span>
              <button type="button" className="sidebar__signout" onClick={logout}>
                sign out
              </button>
            </div>
            <p className="sidebar__note" aria-hidden="true">
              no cloud · no llms <DoodleArrow className="sidebar__note-arrow" />
            </p>
          </div>
        </div>
        <div className="sidebar__torn" aria-hidden="true" />
      </aside>
    </>
  );
}
