import React from 'react';

export function ExternalBenchmarkCard({ program }) {
  const raw = program?.external_benchmarks || program?.benchmark_scores || program?.external_benchmark_scores;
  
  const entries = (() => {
    if (Array.isArray(raw)) return raw;
    if (raw && typeof raw === 'object') {
      return Object.entries(raw)
        .filter(([k]) => !['long_context', 'combined_score', 'benchmark_version', 'scaling'].includes(k))
        .map(([name, score]) => ({ name, score }));
    }
    return [];
  })();

  if (entries.length === 0) return null;

  const formatBenchmarkScore = (score) => {
    if (score == null) return '--';
    const n = Number(score);
    if (!Number.isFinite(n)) return String(score);
    return n > 1 ? n.toFixed(1) : n.toFixed(3);
  };

  return (
    <div className="card" style={{ padding: 12, marginTop: 16 }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600, marginBottom: 10 }}>
        External Benchmarks
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', gap: 12 }}>
        {entries.map((b, i) => (
          <div key={i} style={{ padding: '8px', background: 'var(--bg-secondary)', borderRadius: 4, border: '1px solid var(--border)' }}>
            <div style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={b.name}>
              {b.name.replace(/_/g, ' ')}
            </div>
            <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>
              {formatBenchmarkScore(b.score)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default React.memo(ExternalBenchmarkCard);
