import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext.jsx';
import { useHealth } from '../context/HealthContext.jsx';
import './Login.css';

export function Login() {
  const { login, register } = useAuth();
  const { status } = useHealth();
  const navigate = useNavigate();

  const [mode, setMode] = useState('signin');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const isSignup = mode === 'signup';

  async function handleSubmit(e) {
    e.preventDefault();
    setError('');
    if (!username.trim() || !password) {
      setError('Username and password are required.');
      return;
    }
    if (isSignup) {
      if (username.trim().length < 3) {
        setError('Username must be at least 3 characters.');
        return;
      }
      if (!/^[A-Za-z0-9._-]+$/.test(username.trim())) {
        setError('Username may only contain letters, digits, ".", "_" and "-".');
        return;
      }
      if (password.length < 8) {
        setError('Password must be at least 8 characters.');
        return;
      }
      if (password !== confirm) {
        setError('Passwords do not match.');
        return;
      }
    }
    setBusy(true);
    try {
      if (isSignup) await register(username, password);
      else await login(username, password);
      navigate('/', { replace: true });
    } catch (err) {
      setError(err.detail || err.message || 'Authentication failed.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth">
      <div className="auth__panel">
        <div className="auth__brand">
          <span className="auth__glyph" aria-hidden="true">L</span>
          <h1 className="auth__title">Lexi<span className="auth__title-hl">NLP</span></h1>
          <p className="auth__tagline">
            A field notebook for news — extractive summarization,
            vocabulary learning, and offline listening, kept privately for your account.
          </p>
          <ul className="auth__points">
            <li>Notes and history are visible only to you</li>
            <li>Three classical summarization algorithms</li>
            <li>Runs fully offline — no cloud, no LLMs</li>
          </ul>
          <div className={`auth__health auth__health--${status}`} aria-hidden="true">
            <span className="auth__health-dot" />
            {status === 'online' ? 'API online' : status === 'offline' ? 'API offline' : 'Checking API…'}
          </div>
        </div>

        <div className="auth__form-wrap">
          <form className="auth__form" onSubmit={handleSubmit}>
            <h2 className="auth__form-title">{isSignup ? 'Create your account' : 'Welcome back'}</h2>
            <p className="auth__form-sub">
              {isSignup
                ? 'Pick a username and password — your notes, history, and audio stay separate from everyone else\'s.'
                : 'Sign in to reach your summary archive, vocabulary notes, and listening studio.'}
            </p>

            <label className="field__label" htmlFor="auth-username">Username</label>
            <input
              id="auth-username"
              className="auth__input"
              type="text"
              autoComplete="username"
              value={username}
              onChange={(e) => { setUsername(e.target.value); setError(''); }}
              placeholder={isSignup ? 'e.g. siva_reader' : 'Your username'}
              autoFocus
            />

            <label className="field__label" htmlFor="auth-password">Password</label>
            <input
              id="auth-password"
              className="auth__input"
              type="password"
              autoComplete={isSignup ? 'new-password' : 'current-password'}
              value={password}
              onChange={(e) => { setPassword(e.target.value); setError(''); }}
              placeholder={isSignup ? 'At least 8 characters' : 'Your password'}
            />

            {isSignup ? (
              <>
                <label className="field__label" htmlFor="auth-confirm">Confirm password</label>
                <input
                  id="auth-confirm"
                  className="auth__input"
                  type="password"
                  autoComplete="new-password"
                  value={confirm}
                  onChange={(e) => { setConfirm(e.target.value); setError(''); }}
                  placeholder="Repeat your password"
                />
              </>
            ) : null}

            {error ? <div className="callout-error auth__error" role="alert">{error}</div> : null}

            <button type="submit" className="auth__submit" disabled={busy}>
              {busy ? 'Please wait…' : isSignup ? 'Create account' : 'Sign in'}
            </button>

            <button
              type="button"
              className="auth__switch"
              onClick={() => { setMode(isSignup ? 'signin' : 'signup'); setError(''); }}
            >
              {isSignup ? 'Already have an account? Sign in' : "Don't have an account? Create one"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
