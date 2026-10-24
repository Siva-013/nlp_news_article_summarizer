import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { AppStateProvider } from './context/AppStateContext.jsx';
import { AuthProvider, useAuth } from './context/AuthContext.jsx';
import { HealthProvider } from './context/HealthContext.jsx';
import { ToastProvider } from './components/ui/ToastContext.jsx';
import { Shell } from './components/shell/Shell.jsx';
import { Login } from './views/Login.jsx';
import { Summarizer } from './views/Summarizer.jsx';
import { Vocabulary } from './views/Vocabulary.jsx';
import { Notes } from './views/Notes.jsx';
import { History } from './views/History.jsx';
import { Listen } from './views/Listen.jsx';
import { Evaluate } from './views/Evaluate.jsx';

function AppSplash() {
  return (
    <div className="app-splash" role="status" aria-label="Loading session">
      <span className="app-splash__glyph" aria-hidden="true">L</span>
      <p>Restoring your session…</p>
    </div>
  );
}

function RequireAuth({ children }) {
  const { ready, authenticated } = useAuth();
  if (!ready) return <AppSplash />;
  if (!authenticated) return <Navigate to="/login" replace />;
  return children;
}

export default function App() {
  return (
    <AuthProvider>
      <HealthProvider>
        <ToastProvider>
          <BrowserRouter>
            <AppStateProvider>
              <Routes>
                <Route path="/login" element={<Login />} />
                <Route
                  element={
                    <RequireAuth>
                      <Shell />
                    </RequireAuth>
                  }
                >
                  <Route index element={<Summarizer />} />
                  <Route path="vocabulary" element={<Vocabulary />} />
                  <Route path="notes" element={<Notes />} />
                  <Route path="history" element={<History />} />
                  <Route path="listen" element={<Listen />} />
                  <Route path="evaluate" element={<Evaluate />} />
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Route>
              </Routes>
            </AppStateProvider>
          </BrowserRouter>
        </ToastProvider>
      </HealthProvider>
    </AuthProvider>
  );
}
