import React from 'react';

export function ConfigField({ label, children, description }) {
  return (
    <div className="config-item">
      <label>{label}</label>
      {children}
      {description && <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>{description}</div>}
    </div>
  );
}

export default ConfigField;
