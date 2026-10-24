import './Skeleton.css';

export function SkeletonBlock({ lines = 4, className = '' }) {
  return (
    <div className={`skeleton ${className}`} aria-hidden="true">
      {Array.from({ length: lines }).map((_, i) => (
        <span
          key={i}
          className="skeleton__line"
          style={{ width: `${72 + ((i * 13) % 26)}%`, animationDelay: `${i * 110}ms` }}
        />
      ))}
    </div>
  );
}

export function SkeletonRows({ rows = 4 }) {
  return (
    <div className="skeleton-rows" aria-hidden="true">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="skeleton-rows__row" style={{ animationDelay: `${i * 90}ms` }}>
          <span className="skeleton__circle" />
          <span className="skeleton__line" style={{ width: `${48 + ((i * 11) % 38)}%` }} />
        </div>
      ))}
    </div>
  );
}
