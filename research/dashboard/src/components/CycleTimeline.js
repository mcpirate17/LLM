import { apiCall } from "../services/apiService";
import React, { useCallback, useEffect, useState } from 'react';

const API_BASE = process.env.REACT_APP_API_URL || '';

function formatTs(ts) {
  if (!ts) return 'n/a';
  const ms = ts > 1e12 ? ts : ts * 1000;
  return new Date(ms).toLocaleString();
}

function CycleTimeline() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [modeFilter, setModeFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [query, setQuery] = useState('');
  const [preset, setPreset] = useState('');
  const [recentHours, setRecentHours] = useState(null);

  const fetchHistory = useCallback(async () => {
    try {
      const params = new URLSearchParams();
      params.set('n', '120');
      if (modeFilter) params.set('mode', modeFilter);
      if (statusFilter) params.set('status', statusFilter);
      if (query.trim()) params.set('q', query.trim());
      const res = await apiCall(`/api/aria/cycle-history?${params.toString()}`);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      setRows(Array.isArray(data) ? data : []);
      setError(null);
    } catch (err) {
      setError(err.message || 'Failed to load cycle history');
    } finally {
      setLoading(false);
    }
  }, [modeFilter, statusFilter, query]);

  const exportHistory = useCallback((format) => {
    const params = new URLSearchParams();
    params.set('n', '500');
    if (modeFilter) params.set('mode', modeFilter);
    if (statusFilter) params.set('status', statusFilter);
    if (query.trim()) params.set('q', query.trim());
    params.set('format', format);
    const url = `${API_BASE}/api/aria/cycle-history?${params.toString()}`;

    if (format === 'json') {
      window.open(url, '_blank', 'noopener,noreferrer');
      return;
    }

    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'aria_cycle_history.csv';
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
  }, [modeFilter, statusFilter, query]);

  useEffect(() => {
    fetchHistory();
    const interval = setInterval(fetchHistory, 10000);
    return () => clearInterval(interval);
  }, [fetchHistory]);

  const applyPreset = useCallback((nextPreset) => {
    if (nextPreset === 'failures') {
      setModeFilter('');
      setStatusFilter('failed');
      setQuery('');
      setRecentHours(null);
      setPreset('failures');
      return;
    }
    if (nextPreset === 'evolution') {
      setModeFilter('evolution');
      setStatusFilter('');
      setQuery('');
      setRecentHours(null);
      setPreset('evolution');
      return;
    }
    if (nextPreset === 'last24h') {
      setRecentHours(24);
      setPreset('last24h');
      return;
    }
    setModeFilter('');
    setStatusFilter('');
    setQuery('');
    setRecentHours(null);
    setPreset('');
  }, []);

  const displayedRows = rows.filter((row) => {
    if (!recentHours) return true;
    const ts = row?.timestamp || row?.entry_timestamp;
    if (!ts) return false;
    const rowTsMs = ts > 1e12 ? ts : ts * 1000;
    const cutoff = Date.now() - (recentHours * 60 * 60 * 1000);
    return rowTsMs >= cutoff;
  });

  return (
    <div className="card" style={{ marginBottom: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <div style={{ fontSize: 14, fontWeight: 600 }}>Cycle Timeline</div>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          <button className="refresh-btn" onClick={() => exportHistory('json')}>
            Export JSON
          </button>
          <button className="refresh-btn" onClick={() => exportHistory('csv')}>
            Export CSV
          </button>
          <button className="refresh-btn" onClick={fetchHistory} disabled={loading}>
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
        <button className="refresh-btn" onClick={() => applyPreset('failures')} style={{ opacity: preset === 'failures' ? 1 : 0.8 }}>
          Failures
        </button>
        <button className="refresh-btn" onClick={() => applyPreset('evolution')} style={{ opacity: preset === 'evolution' ? 1 : 0.8 }}>
          Evolution
        </button>
        <button className="refresh-btn" onClick={() => applyPreset('last24h')} style={{ opacity: preset === 'last24h' ? 1 : 0.8 }}>
          Last 24h
        </button>
        <button className="refresh-btn" onClick={() => applyPreset('')}>
          Clear
        </button>
      </div>

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
        <select
          value={modeFilter}
          onChange={(e) => {
            setModeFilter(e.target.value);
            setPreset('');
          }}
          style={{
            background: 'var(--bg-primary)',
            border: '1px solid var(--border)',
            borderRadius: 6,
            color: 'var(--text-primary)',
            fontSize: 12,
            padding: '6px 8px',
          }}
        >
          <option value="">All modes</option>
          <option value="synthesis">synthesis</option>
          <option value="investigation">investigation</option>
          <option value="validation">validation</option>
          <option value="evolution">evolution</option>
          <option value="novelty">novelty</option>
        </select>
        <select
          value={statusFilter}
          onChange={(e) => {
            setStatusFilter(e.target.value);
            setPreset('');
          }}
          style={{
            background: 'var(--bg-primary)',
            border: '1px solid var(--border)',
            borderRadius: 6,
            color: 'var(--text-primary)',
            fontSize: 12,
            padding: '6px 8px',
          }}
        >
          <option value="">All statuses</option>
          <option value="completed">completed</option>
          <option value="failed">failed</option>
        </select>
        <input
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setPreset('');
          }}
          placeholder="Search reasoning or errors..."
          style={{
            flex: 1,
            minWidth: 220,
            background: 'var(--bg-primary)',
            border: '1px solid var(--border)',
            borderRadius: 6,
            color: 'var(--text-primary)',
            fontSize: 12,
            padding: '6px 8px',
          }}
        />
      </div>

      {error && (
        <div className="error-banner" style={{ marginBottom: 10 }}>
          {error}
        </div>
      )}

      {displayedRows.length === 0 && !loading ? (
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          No persisted cycle history yet. Start continuous cycle mode to populate this timeline.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {displayedRows.slice().reverse().map((row, idx) => (
            <div
              key={`${row.entry_id || row.timestamp || idx}`}
              style={{
                border: '1px solid var(--border)',
                borderRadius: 6,
                padding: '8px 10px',
                background: 'var(--bg-tertiary)',
              }}
            >
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', fontSize: 11, color: 'var(--text-muted)' }}>
                <span>Cycle #{row.cycle_index ?? 'n/a'}</span>
                <span>Mode: {row.mode || 'synthesis'}</span>
                <span>Status: {row.status || 'completed'}</span>
                <span>{formatTs(row.timestamp || row.entry_timestamp)}</span>
              </div>
              <div style={{ marginTop: 4, fontSize: 12, color: 'var(--text-secondary)' }}>
                ΔPrograms {row.delta_programs ?? 0} · ΔS1 {row.delta_stage1_survivors ?? 0} · S1 total {row.stage1_survivors ?? 0}
              </div>
              {row.reasoning && (
                <div style={{ marginTop: 4, fontSize: 12, color: 'var(--text-secondary)' }}>
                  {row.reasoning}
                </div>
              )}
              {row.error && (
                <div style={{ marginTop: 4, fontSize: 12, color: 'var(--accent-yellow)' }}>
                  Error: {row.error}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default CycleTimeline;
