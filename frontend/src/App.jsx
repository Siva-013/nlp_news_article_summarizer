import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { AppStateProvider } from './context/AppStateContext.jsx';
import { HealthProvider } from './context/HealthContext.jsx';
import { ToastProvider } from './components/ui/ToastContext.jsx';
import { Shell } from './components/shell/Shell.jsx';
import { Summarizer } from './views/Summarizer.jsx';
import { Vocabulary } from './views/Vocabulary.jsx';
import { Notes } from './views/Notes.jsx';
import { History } from './views/History.jsx';
import { Listen } from './views/Listen.jsx';
import { Evaluate } from './views/Evaluate.jsx';

export default function App() {
  return (
    <AppStateProvider>
      <HealthProvider>
        <ToastProvider>
          <BrowserRouter>
            <Routes>
              <Route element={<Shell />}>
                <Route index element={<Summarizer />} />
                <Route path="vocabulary" element={<Vocabulary />} />
                <Route path="notes" element={<Notes />} />
                <Route path="history" element={<History />} />
                <Route path="listen" element={<Listen />} />
                <Route path="evaluate" element={<Evaluate />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Route>
            </Routes>
          </BrowserRouter>
        </ToastProvider>
      </HealthProvider>
    </AppStateProvider>
  );
}
