import { useState } from 'react';
import { evaluateSummary } from '../api/evaluate.js';
import { Button } from '../components/ui/Button.jsx';
import { Field } from '../components/ui/Field.jsx';
import { EmptyState } from '../components/ui/EmptyState.jsx';
import { useToast } from '../components/ui/ToastContext.jsx';
import { useAppState } from '../context/AppStateContext.jsx';
import './Evaluate.css';

function verdict(f1) {
  if (f1 >= 0.6) return { label: 'Strong overlap', tone: 'success' };
  if (f1 >= 0.35) return { label: 'Moderate overlap', tone: 'info' };
  if (f1 >= 0.15) return { label: 'Weak overlap', tone: 'warning' };
  return { label: 'Little overlap', tone: 'danger' };
}

function ScoreCard({ label, description, score, delayMs }) {
  if (!score) return null;
  const f1 = Number(score.f1) || 0;
  const v = verdict(f1);
  return (
    <div className="card rouge-card" style={{ animationDelay: `${delayMs}ms` }}>
      <div className="rouge-card__head">
        <h3 className="rouge-card__name">{label}</h3>
        <span className={`badge badge--${v.tone}`}>{v.label}</span>
      </div>
      <p className="rouge-card__f1">
        <span className="rouge-card__f1-value mono">{f1.toFixed(4)}</span>
        <span className="rouge-card__f1-label">F1</span>
      </p>
      <div className="meter rouge-card__meter">
        <div className="meter__fill" style={{ width: `${Math.max(2, f1 * 100)}%` }} />
      </div>
      <dl className="rouge-card__pr">
        <div>
          <dt>Precision</dt>
          <dd className="mono">{Number(score.precision).toFixed(4)}</dd>
        </div>
        <div>
          <dt>Recall</dt>
          <dd className="mono">{Number(score.recall).toFixed(4)}</dd>
        </div>
      </dl>
      <p className="rouge-card__desc">{description}</p>
    </div>
  );
}

export function Evaluate() {
  const toast = useToast();
  const { lastSummary } = useAppState();
  const [candidate, setCandidate] = useState(lastSummary?.summary || '');
  const [reference, setReference] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  async function handleEvaluate() {
    if (!candidate.trim() || !reference.trim()) {
      setError('Both candidate and reference summaries are required.');
      return;
    }
    setError('');
    setLoading(true);
    setResult(null);
    try {
      const res = await evaluateSummary({
        candidate_summary: candidate.trim(),
        reference_summary: reference.trim(),
      });
      setResult(res);
    } catch (err) {
      setError(err.detail || err.message || 'Evaluation failed.');
      toast('ROUGE evaluation failed', 'error');
    } finally {
      setLoading(false);
    }
  }

  function handleSwap() {
    setCandidate(reference);
    setReference(candidate);
    setResult(null);
  }

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <p className="eyebrow">Quantitative evaluation</p>
          <h1 className="page-head__title">ROUGE <span className="text-gradient">scores</span></h1>
          <p className="page-head__sub">
            Compare a generated summary against a human reference with ROUGE-1, ROUGE-2 (bigram),
            and ROUGE-L (longest common subsequence) — all computed by the backend from scratch.
          </p>
        </div>
      </header>

      <div className="rouge-grid">
        <div className="card rouge-pane">
          <div className="card__head">
            <span className="card__title">Candidate <span className="rouge-pane__tag mono">system output</span></span>
            <Button variant="ghost" size="sm" disabled={!lastSummary?.summary} onClick={() => setCandidate(lastSummary.summary)}>
              Use last summary
            </Button>
          </div>
          <div className="rouge-pane__body">
            <Field
              id="candidate"
              multiline
              rows={8}
              value={candidate}
              onChange={(e) => { setCandidate(e.target.value); setError(''); }}
              onClear={() => setCandidate('')}
              placeholder="The summary you want to score…"
            />
          </div>
        </div>

        <div className="rouge-pane__swap">
          <Button variant="secondary" size="sm" onClick={handleSwap} title="Swap candidate and reference" disabled={!candidate && !reference}>
            ⇄ swap
          </Button>
        </div>

        <div className="card rouge-pane">
          <div className="card__head">
            <span className="card__title">Reference <span className="rouge-pane__tag mono">human ground truth</span></span>
          </div>
          <div className="rouge-pane__body">
            <Field
              id="reference"
              multiline
              rows={8}
              value={reference}
              onChange={(e) => { setReference(e.target.value); setError(''); }}
              onClear={() => setReference('')}
              placeholder="The reference summary to compare against…"
              error={error}
            />
          </div>
        </div>
      </div>

      <div className="rouge-run">
        <Button variant="primary" size="lg" loading={loading} disabled={!candidate.trim() || !reference.trim()} onClick={handleEvaluate}>
          {loading ? 'Computing…' : 'Compute ROUGE'}
        </Button>
      </div>

      <section aria-label="ROUGE results" aria-live="polite">
        {!result && !loading ? (
          <EmptyState
            icon="％"
            title="No scores yet"
            description="Paste a candidate and a reference summary, then compute. Precision, recall, and F1 appear for each metric."
          />
        ) : null}

        {result ? (
          <div className="rouge-scores">
            <ScoreCard
              label="ROUGE-1"
              description="Unigram overlap — how many individual words the summaries share."
              score={result.rouge1}
              delayMs={0}
            />
            <ScoreCard
              label="ROUGE-2"
              description="Bigram overlap — rewards consecutive word sequences, not just shared vocabulary."
              score={result.rouge2}
              delayMs={90}
            />
            <ScoreCard
              label="ROUGE-L"
              description="Longest common subsequence — structural similarity that ignores word order gaps."
              score={result.rougeL}
              delayMs={180}
            />
          </div>
        ) : null}
      </section>
    </div>
  );
}
