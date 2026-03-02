import React from 'react';

export function FingerprintDiagnosticsCard({ diagnostics }) {
  if (!diagnostics) return null;

  const total = Number(diagnostics.total || 0);
  const byReason = diagnostics.by_reason && typeof diagnostics.by_reason === 'object'
    ? diagnostics.by_reason
    : {};
  const topReasons = Object.entries(byReason)
    .sort((a, b) => Number(b[1] || 0) - Number(a[1] || 0))
    .slice(0, 3);

  return (
    <div className="card">
      <div className="card-title">Fingerprint Diagnostics</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Runtime telemetry for skipped sensitivity probes during fingerprint analysis.
      </p>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 6 }}>
        <strong style={{ color: total > 0 ? 'var(--accent-yellow)' : 'var(--accent-green)' }}>
          Sensitivity skips:
        </strong>{' '}
        {total}
      </div>
      {topReasons.length > 0 && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          Top reasons: {topReasons.map(([reason, count]) => `${reason} (${count})`).join(' · ')}
        </div>
      )}
    </div>
  );
}

export default FingerprintDiagnosticsCard;
