import React, { useEffect } from 'react';

const KeyboardShortcuts = ({ onClose }) => {
  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  const shortcuts = [
    { key: 'Del / Backspace', desc: 'Remove selected node' },
    { key: 'Ctrl + Z', desc: 'Undo' },
    { key: 'Ctrl + Shift + Z', desc: 'Redo' },
    { key: 'Ctrl + S', desc: 'Save workflow' },
    { key: 'Ctrl + Enter', desc: 'Compile + Run' },
    { key: 'Arrow Keys', desc: 'Nudge selected nodes' },
    { key: 'Shift + Arrows', desc: 'Nudge faster' },
    { key: 'Space', desc: 'Pan mode (hold)' },
    { key: '?', desc: 'Toggle this help' },
  ];

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content shortcuts-modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Keyboard Shortcuts</h2>
          <button type="button" className="close-btn" onClick={onClose}>&times;</button>
        </div>
        <div className="shortcuts-grid">
          {shortcuts.map(s => (
            <div key={s.key} className="shortcut-row">
              <span className="shortcut-key">{s.key}</span>
              <span className="shortcut-desc">{s.desc}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default KeyboardShortcuts;
