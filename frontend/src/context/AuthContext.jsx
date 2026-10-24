import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { getMe, login as apiLogin, register as apiRegister } from '../api/auth.js';
import { getStoredToken, storeToken } from '../api/client.js';

/**
 * Session state for the authenticated app.
 * - On mount, validates any stored JWT against /auth/me.
 * - login()/register() store the token and set the user.
 * - Listens for 'lexica:unauthorized' (dispatched by client.js on any 401)
 *   and signs the user out.
 */
const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const token = getStoredToken();
      if (!token) {
        setReady(true);
        return;
      }
      try {
        const me = await getMe();
        if (!cancelled) setUser({ id: me.id, username: me.username, created_at: me.created_at });
      } catch {
        storeToken(null);
      } finally {
        if (!cancelled) setReady(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const onUnauthorized = () => setUser(null);
    window.addEventListener('lexica:unauthorized', onUnauthorized);
    return () => window.removeEventListener('lexica:unauthorized', onUnauthorized);
  }, []);

  const login = useCallback(async (username, password) => {
    const res = await apiLogin({ username: username.trim(), password });
    storeToken(res.access_token);
    setUser({ id: res.user.id, username: res.user.username, created_at: res.user.created_at });
    return res.user;
  }, []);

  const register = useCallback(async (username, password) => {
    const created = await apiRegister({ username: username.trim(), password });
    const res = await apiLogin({ username: username.trim(), password });
    storeToken(res.access_token);
    setUser({ id: res.user.id, username: res.user.username, created_at: res.user.created_at });
    return created;
  }, []);

  const logout = useCallback(() => {
    storeToken(null);
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({ user, ready, authenticated: Boolean(user), login, register, logout }),
    [user, ready, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be inside AuthProvider');
  return ctx;
}
