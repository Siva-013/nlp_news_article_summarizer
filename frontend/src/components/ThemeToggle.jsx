import { useState } from 'react';
import './ThemeToggle.css';

/**
 * Day/night theme switch, shared by the app top strip and the login screen.
 * variants: strip (compact rectangle in the top strip), floating (round,
 * for standalone pages like login).
 */
export function ThemeToggle({ variant = 'strip' }) {
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme || 'day');

  function toggle() {
    const next = theme === 'day' ? 'night' : 'day';
    setTheme(next);
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem('lexica_theme', next);
    } catch {
      /* storage unavailable */
    }
  }

  return (
    <button
      type="button"
      className={`theme-toggle theme-toggle--${variant}`}
      onClick={toggle}
      aria-label={theme === 'day' ? 'Switch to night theme' : 'Switch to day theme'}
      title={theme === 'day' ? 'Night notebook' : 'Day notebook'}
    >
      <span aria-hidden="true">{theme === 'day' ? '☾' : '☀'}</span>
    </button>
  );
}
