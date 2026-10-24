import { Outlet } from 'react-router-dom';
import { TopBar } from './TopBar.jsx';
import { MobileTabBar } from './MobileTabBar.jsx';
import './Shell.css';

export function Shell() {
  return (
    <div className="shell">
      <a className="shell__skip" href="#main-content">Skip to content</a>
      <TopBar />
      <main className="shell__content" id="main-content">
        <Outlet />
      </main>
      <footer className="shell__footer">
        <span>Lexica · extractive NLP study console</span>
        <span className="shell__footer-sep" aria-hidden="true">·</span>
        <span>frequency / tf-idf / textrank</span>
        <span className="shell__footer-sep" aria-hidden="true">·</span>
        <span>no LLMs, no cloud</span>
      </footer>
      <MobileTabBar />
    </div>
  );
}
