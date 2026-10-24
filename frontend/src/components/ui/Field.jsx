import './Field.css';

/**
 * Labeled text input / textarea with optional clear button, hint and error.
 * Paper-toned fields with ink rules; label is a mono uppercase tag.
 */
export function Field({
  id,
  label,
  hint,
  error,
  multiline = false,
  rows = 4,
  value,
  onClear,
  className = '',
  ...rest
}) {
  const hasValue = value != null && String(value).length > 0;
  return (
    <div className={`field ${error ? 'field--error' : ''} ${className}`}>
      <div className="field__top">
        {label ? (
          <label className="field__label" htmlFor={id}>{label}</label>
        ) : (
          <span />
        )}
        {hasValue && onClear ? (
          <button type="button" className="field__clear" onClick={onClear} aria-label={`Clear ${label || 'field'}`}>
            clear ✕
          </button>
        ) : null}
      </div>

      {multiline ? (
        <textarea id={id} className="field__input field__input--area" rows={rows} value={value} {...rest} />
      ) : (
        <input id={id} className="field__input" value={value} {...rest} />
      )}

      {error ? (
        <p className="field__error" role="alert">{error}</p>
      ) : hint ? (
        <p className="field__hint">{hint}</p>
      ) : null}
    </div>
  );
}
