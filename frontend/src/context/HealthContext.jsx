import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { checkHealth } from '../api/health.js';

/**
 * Polls the backend /health endpoint and exposes connection state
 * so the shell can show a live status pill and views can explain
 * network failures clearly.
 */
const HealthContext = createContext(null);

const POLL_MS = 20000;

export function HealthProvider({ children }) {
  const [state, setState] = useState({ status: 'checking', database: false, since: null });

  const refresh = useCallback(async () => {
    try {
      const res = await checkHealth();
      const ok = res.status === 'healthy';
      setState((prev) => ({
        status: ok ? 'online' : 'degraded',
        database: Boolean(res.database_connected),
        since: prev.since === ok ? prev.since : Date.now(),
      }));
    } catch {
      setState({ status: 'offline', database: false, since: null });
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, POLL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  const value = useMemo(
    () => ({ ...state, online: state.status === 'online', refresh }),
    [state, refresh],
  );

  return <HealthContext.Provider value={value}>{children}</HealthContext.Provider>;
}

export function useHealth() {
  const ctx = useContext(HealthContext);
  if (!ctx) throw new Error('useHealth must be inside HealthProvider');
  return ctx;
}
