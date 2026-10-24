import { createContext, useContext, useMemo, useState } from 'react';

/**
 * In-memory workflow state shared across sections.
 * Does not survive a full page reload — SQLite persistence lives on the backend.
 */
const AppStateContext = createContext(null);

export function AppStateProvider({ children }) {
  const [lastArticleText, setLastArticleText] = useState('');
  const [lastSummary, setLastSummary] = useState(null);
  const [lastVocabResult, setLastVocabResult] = useState(null);
  const [lastDetectResult, setLastDetectResult] = useState(null);
  const [ttsPendingText, setTtsPendingText] = useState('');
  const [pendingNote, setPendingNote] = useState(null);
  const [lastHistoryId, setLastHistoryId] = useState(null);

  const value = useMemo(
    () => ({
      lastArticleText,
      setLastArticleText,
      lastSummary,
      setLastSummary,
      lastVocabResult,
      setLastVocabResult,
      lastDetectResult,
      setLastDetectResult,
      ttsPendingText,
      setTtsPendingText,
      pendingNote,
      setPendingNote,
      lastHistoryId,
      setLastHistoryId,
    }),
    [
      lastArticleText,
      lastSummary,
      lastVocabResult,
      lastDetectResult,
      ttsPendingText,
      pendingNote,
      lastHistoryId,
    ],
  );

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState() {
  const ctx = useContext(AppStateContext);
  if (!ctx) throw new Error('useAppState must be inside AppStateProvider');
  return ctx;
}
