import React, { useMemo } from 'react';

/**
 * SimilarityHeatmap — Task 3H
 * 
 * Visualizes a 10x10 CKA similarity matrix comparing:
 * - Top 10 architectures (by composite_score)
 * - 10 most recent reference architectures
 */
export function SimilarityHeatmap({ leaderboardEntries }) {
  const top10 = useMemo(() => {
    return leaderboardEntries
      .filter(e => !e.is_reference)
      .sort((a, b) => (b.composite_score || 0) - (a.composite_score || 0))
      .slice(0, 10);
  }, [leaderboardEntries]);

  const refs10 = useMemo(() => {
    return leaderboardEntries
      .filter(e => e.is_reference)
      .sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0))
      .slice(0, 10);
  }, [leaderboardEntries]);

  if (top10.length === 0 || refs10.length === 0) {
    return (
      <div className="empty-state" style={{ padding: 20 }}>
        <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>
          Insufficient data for similarity heatmap (need candidates and references).
        </div>
      </div>
    );
  }

  const hasMeasuredMatrix = false;

  if (!hasMeasuredMatrix) {
    return (
      <div className="empty-state" style={{ padding: 20 }}>
        <div style={{ color: 'var(--text-primary)', fontSize: 12, fontWeight: 600, marginBottom: 6 }}>
          Similarity matrix unavailable
        </div>
        <div style={{ color: 'var(--text-muted)', fontSize: 12, lineHeight: 1.5 }}>
          The dashboard has candidate and reference rows, but no measured candidate-vs-reference CKA matrix.
          Rendering heuristic similarity here would be fake analytics, so this view stays disabled until the API exposes real pairwise data.
        </div>
      </div>
    );
  }

  const CELL_SIZE = 30;
  const LABEL_WIDTH = 100;
  const LABEL_HEIGHT = 80;
  const W = LABEL_WIDTH + refs10.length * CELL_SIZE + 20;
  const H = LABEL_HEIGHT + top10.length * CELL_SIZE + 20;

  const getColor = (val) => {
    // 0 (blue) -> 1 (red/purple)
    const opacity = 0.2 + 0.8 * val;
    return `rgba(138, 92, 248, ${opacity})`; // purple accent
  };

  return (
    <div style={{ padding: 10, overflow: 'auto' }}>
      <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
        CKA Similarity: Top Candidates vs References
      </div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
        {/* X Labels (References) */}
        {refs10.map((ref, i) => (
          <text
            key={i}
            x={LABEL_WIDTH + i * CELL_SIZE + CELL_SIZE / 2}
            y={LABEL_HEIGHT - 5}
            transform={`rotate(-45, ${LABEL_WIDTH + i * CELL_SIZE + CELL_SIZE / 2}, ${LABEL_HEIGHT - 5})`}
            fontSize={9}
            fill="var(--text-muted)"
            textAnchor="start"
          >
            {ref.reference_name || ref.architecture_family?.slice(0, 12)}
          </text>
        ))}

        {/* Y Labels (Candidates) */}
        {top10.map((cand, j) => (
          <text
            key={j}
            x={LABEL_WIDTH - 5}
            y={LABEL_HEIGHT + j * CELL_SIZE + CELL_SIZE / 2 + 4}
            fontSize={9}
            fill="var(--text-muted)"
            textAnchor="end"
          >
            {cand.result_id?.slice(0, 8)}
          </text>
        ))}

        {/* Matrix Cells */}
        {matrix.map((row, j) => 
          row.map((val, i) => (
            <rect
              key={`${i}-${j}`}
              x={LABEL_WIDTH + i * CELL_SIZE}
              y={LABEL_HEIGHT + j * CELL_SIZE}
              width={CELL_SIZE - 2}
              height={CELL_SIZE - 2}
              fill={getColor(val)}
              rx={2}
            >
              <title>{`Sim: ${val.toFixed(3)}\nCand: ${top10[j].result_id}\nRef: ${refs10[i].reference_name || refs10[i].architecture_family}`}</title>
            </rect>
          ))
        )}
      </svg>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 10, fontSize: 10, color: 'var(--text-muted)' }}>
        <span>Similarity:</span>
        <div style={{ display: 'flex', gap: 2 }}>
          {[0.2, 0.4, 0.6, 0.8, 1.0].map(v => (
            <div key={v} style={{ width: 12, height: 12, background: getColor(v), borderRadius: 2 }} />
          ))}
        </div>
        <span>Low → High</span>
      </div>
    </div>
  );
}

export default SimilarityHeatmap;
