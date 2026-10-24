import { useEffect } from 'react';
import { Button } from './Button.jsx';
import './ConfirmDialog.css';

export function ConfirmDialog({ open, title, description, confirmLabel = 'Delete', onCancel, onConfirm }) {
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') onCancel?.();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onCancel]);

  if (!open) return null;

  return (
    <div className="dialog-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onCancel?.(); }}>
      <div className="dialog" role="alertdialog" aria-modal="true" aria-label={title}>
        <div className="dialog__icon" aria-hidden="true">!</div>
        <h3 className="dialog__title">{title}</h3>
        {description ? <p className="dialog__desc">{description}</p> : null}
        <div className="dialog__actions">
          <Button variant="ghost" onClick={onCancel}>Cancel</Button>
          <Button variant="danger" onClick={onConfirm}>{confirmLabel}</Button>
        </div>
      </div>
    </div>
  );
}
