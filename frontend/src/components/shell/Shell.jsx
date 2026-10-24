import { useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { Sidebar } from './Sidebar.jsx';
import { TopStrip } from './TopStrip.jsx';
import { MobileTabBar } from './MobileTabBar.jsx';
import './Shell.css';

const SECTION_TITLES = {
  '/': 'Analyze',
  '/vocabulary': 'Learn',
  '/notes': 'Learn',
  '/history': 'Review',
  '/listen': 'Listen',
  '/evaluate': 'Review',
};

export function Shell() {
  const [navOpen, setNavOpen] = useState(false);
  const { pathname } = useLocation();

  const crumbs =
    SECTION_TITLES[pathname] === 'Analyze'
      ? ['Summarize Article']
      : SECTION_TITLES[pathname] === 'Learn'
        ? [pathname === '/vocabulary' ? 'Vocabulary' : 'Notes']
        : SECTION_TITLES[pathname] === 'Review'
          ? [pathname === '/history' ? 'History' : 'ROUGE Evaluation']
          : ['Listen Studio'];

  return (
    <div className="shell">
      <a className="shell__skip" href="#main-content">Skip to content</a>
      <Sidebar open={navOpen} onClose={() => setNavOpen(false)} />
      <div className="shell__main">
        <TopStrip section={SECTION_TITLES[pathname] || 'Analyze'} crumbs={crumbs} onMenuOpen={() => setNavOpen(true)} />
        <main className="shell__content" id="main-content">
          <Outlet />
        </main>
        <footer className="shell__folio">
          <span>LexiNLP · a field notebook for news</span>
          <span aria-hidden="true">·</span>
          <span>frequency / tf-idf / textrank</span>
          <span aria-hidden="true">·</span>
          <span>printed locally, no cloud</span>
        </footer>
      </div>
      <MobileTabBar />
    </div>
  );
}
