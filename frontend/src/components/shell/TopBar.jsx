import { NavLink } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext.jsx';
import { useHealth } from '../../context/HealthContext.jsx';
import './TopBar.css';

export const NAV_ITEMS = [
  { to: '/', label: 'Summarize', icon: '⌗', end: true },
  { to: '/vocabulary', label: 'Vocabulary', icon: 'Aa' },
  { to: '/notes', label: 'Notes', icon: '✎' },
  { to: '/history', label: 'History', icon: '⛭' },
  { to: '/listen', label: 'Listen', icon: '▶' },
  { to: '/evaluate', label: 'Evaluate', icon: '％' },
];

function HealthPill() {
  const { status, online, refresh } = useHealth();
  const label =
    status === 'online' ? 'API online' : status === 'degraded' ? 'Degraded' : status === 'checking' ? 'Checking…' : 'API offline';
  return (
    <button
      type="button"
      className={`health-pill health-pill--${status}`}
      onClick={refresh}
      title="Backend connection status — click to re-check"
    >
      <span className="health-pill__dot" aria-hidden="true" />
      <span className="health-pill__label">{label}</span>
    </button>
  );
}

function UserChip() {
  const { user, logout } = useAuth();
  if (!user) return null;
  const initial = (user.username || '?').charAt(0).toUpperCase();
  return (
    <div className="user-chip">
      <span className="user-chip__avatar" aria-hidden="true">{initial}</span>
      <span className="user-chip__name">{user.username}</span>
      <button type="button" className="user-chip__signout" onClick={logout} title="Sign out">
        Sign out
      </button>
    </div>
  );
}

export function TopBar() {
  return (
    <header className="topbar">
      <div className="topbar__inner">
        <NavLink to="/" className="topbar__brand">
          <span className="topbar__glyph" aria-hidden="true">
            <span className="topbar__glyph-core">L</span>
          </span>
          <span className="topbar__wordmark">
            Lexica
            <span className="topbar__tag">NLP study console</span>
          </span>
        </NavLink>

        <nav className="topbar__nav" aria-label="Primary">
          {NAV_ITEMS.map(({ to, label, icon, end }) => (
            <NavLink key={to} to={to} end={end} className={({ isActive }) => `topbar__link ${isActive ? 'topbar__link--active' : ''}`}>
              <span className="topbar__link-icon" aria-hidden="true">{icon}</span>
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="topbar__right">
          <HealthPill />
          <UserChip />
        </div>
      </div>
    </header>
  );
}
