import React, { useState, useEffect, useCallback } from 'react';
import { apiCall } from '../../services/apiService';

export function LineagePanel({ resultId }) {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadLineage = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // For now, fetch recent designer runs. 
      // Ideally this would be a real tree-search API for parent/child relationships.
      const res = await apiCall(`/api/designer/lineage?limit=20`);
      const data = await res.json().catch(() => []);
      if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`);
      setRows(Array.isArray(data) ? data : []);
    } catch (err) {
      setError(err?.message || String(err));
      setRows([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadLineage();
  }, [loadLineage, resultId]);

  return (
    <div style={{ padding: '10px 14px', borderTop: '1px solid var(--border)', maxHeight: 300, overflowY: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>
          Evolution Lineage
        </span>
        <button
          onClick={loadLineage}
          disabled={loading}
          style={{
            marginLeft: 'auto',
            fontSize: 10,
            padding: '2px 8px',
            borderRadius: 4,
            border: '1px solid var(--border)',
            background: 'none',
            color: 'var(--text-secondary)',
            cursor: 'pointer',
          }}
        >
          {loading ? '...' : 'Refresh'}
        </button>
      </div>

      <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12 }}>
        Tracking recent mutations and graph refinements from the Designer.
      </p>

      {error && <div style={{ fontSize: 11, color: 'var(--accent-red)', marginBottom: 8 }}>{error}</div>}
      
      {!loading && !error && rows.length === 0 && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'center', padding: 12 }}>
          No evolution history found.
        </div>
      )}

      {!loading && !error && rows.map((row, idx) => (
        <div
          key={row.run_id || idx}
          style={{
            borderLeft: `2px solid ${row.status === 'success' ? 'var(--accent-green)' : 'var(--border)'}`,
            padding: '8px 10px',
            marginBottom: 8,
            background: 'var(--bg-secondary)',
            borderRadius: '0 6px 6px 0',
            fontSize: 11,
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{row.run_id?.slice(0, 12) || 'unknown_run'}</span>
            <span style={{ 
              fontSize: 9, 
              padding: '1px 5px', 
              borderRadius: 4, 
              background: row.status === 'success' ? 'rgba(63, 185, 80, 0.15)' : 'var(--bg-tertiary)',
              color: row.status === 'success' ? 'var(--accent-green)' : 'var(--text-muted)'
            }}>
              {row.status?.toUpperCase() || 'IDLE'}
            </span>
          </div>
          <div style={{ color: 'var(--text-secondary)', fontSize: 10 }}>
            Workflow: <span style={{ fontFamily: 'monospace' }}>{row.workflow_id?.slice(0, 12) || '-'}</span>
          </div>
          <div style={{ color: 'var(--text-muted)', fontSize: 10, marginTop: 2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            Fingerprint: {row.graph_fingerprint || '-'}
          </div>
        </div>
      ))}
    </div>
  );
}

export default LineagePanel;
