import './StatusBadge.css';

/** Small printed label. tones: neutral, accent(lime), success, warning, danger, info */
export function StatusBadge({ tone = 'neutral', children, className = '' }) {
  return <span className={`badge badge--${tone} ${className}`}>{children}</span>;
}

/** POS tag chip used in vocabulary views */
export function PosBadge({ pos }) {
  if (!pos) return null;
  const known = ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV'].includes(pos);
  return (
    <span className={`pos-badge ${known ? '' : 'pos-badge--other'}`} title={`Part of speech: ${pos}`}>
      {pos}
    </span>
  );
}
