import { apiCall } from "../services/apiService";
import React, { useState, useEffect } from 'react';
import { DESIGNER_API_BASE } from '../config';

const ImportDialog = ({ onImport, onClose }) => {
  const [survivors, setSurvivors] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [query, setQuery] = useState('');
  const [sortBy, setSortBy] = useState('loss_ratio');
  const [limit, setLimit] = useState(25);
  const [minNovelty, setMinNovelty] = useState(0);
  const [importingId, setImportingId] = useState(null);

  const fetchSurvivors = async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        n: String(limit),
        sort_by: String(sortBy),
        min_novelty: String(minNovelty),
      });
      const res = await apiCall(`/api/v1/import/survivors?${params.toString()}`);
      const data = await res.json();
      // Support both proxy envelope ({ survivors: [...] }) and direct list payload.
      const rows = Array.isArray(data) ? data : (data?.survivors || []);
      setSurvivors(Array.isArray(rows) ? rows : []);
    } catch (_) {
      setError('Failed to fetch research survivors');
      setSurvivors([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSurvivors();
  }, [limit, sortBy, minNovelty]);

  const handleImport = async (resultId) => {
    setImportingId(resultId);
    setError(null);
    try {
      const res = await apiCall(`/api/v1/import/survivors/${encodeURIComponent(resultId)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ result_id: resultId }),
      });
      const data = await res.json();
      // Support both legacy envelope and direct workflow response.
      const importedWorkflow = data?.workflow || (data?.nodes && data?.edges ? data : null);
      if (importedWorkflow) {
        onImport(importedWorkflow);
        onClose();
      } else {
        setError('Import failed: ' + (data?.error || data?.detail || 'Unknown error'));
      }
    } catch (e) {
      setError('Import failed: ' + e.message);
    } finally {
      setImportingId(null);
    }
  };

  const filtered = survivors.filter((s) => {
    const q = query.trim().toLowerCase();
    if (!q) return true;
    const rid = String(s?.result_id || '').toLowerCase();
    const name = String(s?.name || '').toLowerCase();
    const fp = String(s?.graph_fingerprint || '').toLowerCase();
    return rid.includes(q) || name.includes(q) || fp.includes(q);
  });

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content import-modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Import from AI Scientist</h2>
          <button className="close-btn" onClick={onClose}>&times;</button>
        </div>
        <div className="import-controls">
          <input
            className="import-search"
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by result id, name, fingerprint"
          />
          <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
            <option value="loss_ratio">Sort: loss ratio</option>
            <option value="novelty_score">Sort: novelty</option>
            <option value="created_at">Sort: newest</option>
          </select>
          <label>
            min novelty
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={minNovelty}
              onChange={(e) => setMinNovelty(Math.max(0, Math.min(1, Number(e.target.value) || 0)))}
            />
          </label>
          <label>
            limit
            <input
              type="number"
              min={5}
              max={100}
              step={5}
              value={limit}
              onChange={(e) => setLimit(Math.max(5, Math.min(100, Number(e.target.value) || 25)))}
            />
          </label>
          <button className="primary small" onClick={fetchSurvivors} disabled={loading}>Refresh</button>
        </div>
        {loading && <div className="loading-state">Fetching survivors...</div>}
        {error && <div className="error-state">{error}</div>}

        <div className="survivor-list">
          {!loading && filtered.length === 0 && <p className="muted">No survivors match this filter.</p>}
          {filtered.map(s => (
            <div key={s.result_id} className="survivor-item">
              <div className="survivor-info">
                <strong>{s.name || s.result_id}</strong>
                <div className="survivor-meta">
                  Loss: {Number.isFinite(s.loss_ratio) ? s.loss_ratio.toFixed(4) : '-'}
                  {' · '}
                  Novelty: {Number.isFinite(s.novelty_score) ? s.novelty_score.toFixed(2) : '-'}
                  {s.graph_fingerprint ? ` · ${String(s.graph_fingerprint).slice(0, 12)}` : ''}
                </div>
              </div>
              <button
                className="primary small"
                onClick={() => handleImport(s.result_id)}
                disabled={importingId === s.result_id}
              >
                {importingId === s.result_id ? 'Importing...' : 'Import'}
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ImportDialog;
