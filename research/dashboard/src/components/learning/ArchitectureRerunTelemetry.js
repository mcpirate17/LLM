import React from 'react';

export function ArchitectureRerunTelemetry({ telemetry }) {
  if (!telemetry) return null;

  const uniqueCount = Number(telemetry.unique_fingerprint_count || 0);
  const totalRows = Number(telemetry.total_result_rows || 0);
  const repeatRows = Number(telemetry.repeat_result_rows || 0);
  const rerunRatio = Number(telemetry.rerun_ratio || 0);
  const topConcentration = Number(telemetry.top_fingerprint_concentration || 0);
  const weightingMode = telemetry.weighting_mode || 'unknown';

  return (
    <div className="card">
      <div className="card-title">Unique Architectures vs Reruns</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Breadth telemetry for architecture search. High rerun ratios or high top-fingerprint concentration
        indicate learning signal is coming from repeated identities rather than broad exploration.
      </p>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 8 }}>
        <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          <strong style={{ color: 'var(--accent-green)' }}>Unique fingerprints:</strong> {uniqueCount}
        </span>
        <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          <strong style={{ color: 'var(--text-muted)' }}>Rows:</strong> {totalRows}
        </span>
        <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          <strong style={{ color: rerunRatio >= 0.6 ? 'var(--accent-yellow)' : 'var(--text-muted)' }}>Rerun ratio:</strong>{' '}
          {(rerunRatio * 100).toFixed(1)}%
        </span>
        <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          <strong style={{ color: topConcentration >= 0.35 ? 'var(--accent-yellow)' : 'var(--text-muted)' }}>Top fingerprint concentration:</strong>{' '}
          {(topConcentration * 100).toFixed(1)}%
        </span>
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
        Repeat rows: {repeatRows} · Weighting mode: {weightingMode}
      </div>
    </div>
  );
}

export default ArchitectureRerunTelemetry;
