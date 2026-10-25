/**
 * Hand-drawn SVG doodles — marker underlines, scribbles, arrows.
 * Roughly sketched paths with round caps so they read as pen strokes,
 * not geometric shapes. All accept a className for positioning.
 */

const LIME = '#C6F24E';
const INK = 'var(--ink)';

export function MarkerUnderline({ className = '', color = LIME, flip = false }) {
  return (
    <svg
      className={`doodle ${className}`}
      viewBox="0 0 220 14"
      preserveAspectRatio="none"
      aria-hidden="true"
      style={{ transform: flip ? 'scaleX(-1)' : undefined }}
    >
      <path
        d="M4 9 C 40 3, 78 11, 112 7 S 178 4, 216 8"
        fill="none"
        stroke={color}
        strokeWidth="7"
        strokeLinecap="round"
        opacity="0.85"
        pathLength="1"
        className="doodle-draw"
      />
    </svg>
  );
}

export function CircleScribble({ className = '', color = LIME }) {
  return (
    <svg className={`doodle ${className}`} viewBox="0 0 120 64" aria-hidden="true">
      <path
        d="M18 34 C 14 12, 44 4, 66 6 C 96 8, 112 22, 108 38 C 104 54, 74 62, 48 58 C 24 55, 8 46, 14 28 C 18 16, 36 8, 58 8"
        fill="none"
        stroke={color}
        strokeWidth="5"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function DoodleArrow({ className = '', color = LIME }) {
  return (
    <svg className={`doodle ${className}`} viewBox="0 0 90 70" aria-hidden="true">
      <path
        d="M8 8 C 26 30, 48 44, 74 50 M74 50 l -14 -2 M74 50 l -5 -13"
        fill="none"
        stroke={color}
        strokeWidth="5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function Squiggle({ className = '', color = INK }) {
  return (
    <svg className={`doodle ${className}`} viewBox="0 0 120 18" preserveAspectRatio="none" aria-hidden="true">
      <path
        d="M4 12 q 10 -12, 20 0 t 20 0 t 20 0 t 20 0 t 20 0"
        fill="none"
        stroke={color}
        strokeWidth="3"
        strokeLinecap="round"
        opacity="0.55"
      />
    </svg>
  );
}

export function Starburst({ className = '', size = 22, color = LIME }) {
  const spikes = 8;
  const outer = 10;
  const inner = 4.4;
  const pts = [];
  for (let i = 0; i < spikes * 2; i++) {
    const r = i % 2 === 0 ? outer : inner;
    const a = (Math.PI * i) / spikes - Math.PI / 2;
    pts.push(`${(11 + r * Math.cos(a)).toFixed(1)},${(11 + r * Math.sin(a)).toFixed(1)}`);
  }
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 22 22" aria-hidden="true">
      <polygon points={pts.join(' ')} fill={color} stroke={INK} strokeWidth="1.4" strokeLinejoin="round" />
    </svg>
  );
}
