import { NavLink } from 'react-router-dom';
import { NAV_ITEMS } from './Sidebar.jsx';
import './MobileTabBar.css';

export function MobileTabBar() {
  return (
    <nav className="tabbar" aria-label="Primary (mobile)">
      {NAV_ITEMS.map(({ to, label, icon, end }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          className={({ isActive }) => `tabbar__item ${isActive ? 'tabbar__item--active' : ''}`}
        >
          <span className="tabbar__icon" aria-hidden="true">{icon}</span>
          <span className="tabbar__label">{label}</span>
        </NavLink>
      ))}
    </nav>
  );
}
