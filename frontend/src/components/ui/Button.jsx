import './Button.css';

/**
 * Shared button — flat editorial style.
 * variants: primary (ink slab, lime edge), secondary (paper + rule),
 * ghost (transparent), danger (rust ink)
 */
export function Button({
  variant = 'secondary',
  size = 'md',
  loading = false,
  disabled = false,
  type = 'button',
  className = '',
  children,
  ...rest
}) {
  return (
    <button
      type={type}
      className={`btn btn--${variant} btn--${size} ${loading ? 'btn--loading' : ''} ${className}`}
      disabled={disabled || loading}
      {...rest}
    >
      {loading ? (
        <span className="btn__spinner" aria-hidden="true">
          <span />
        </span>
      ) : null}
      <span className="btn__label">{children}</span>
    </button>
  );
}
