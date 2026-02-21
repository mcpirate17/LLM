import React, { useState, useEffect } from 'react';

const DESIGNER_API_BASE = import.meta.env.VITE_DESIGNER_API_BASE || 'http://127.0.0.1:5000';

const ImportDialog = ({ onImport, onClose }) => {
  const [survivors, setSurvivors] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchSurvivors = async () => {
      setLoading(true);
      try {
        const res = await fetch(`${DESIGNER_API_BASE}/api/designer/import/survivors`);
        const data = await res.json();
        setSurvivors(data.survivors || []);
      } catch (err) {
        setError('Failed to fetch research survivors');
      } finally {
        setLoading(false);
      }
    };
    fetchSurvivors();
  }, []);

  const handleImport = async (resultId) => {
    try {
      const res = await fetch(`${DESIGNER_API_BASE}/api/designer/import`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ result_id: resultId }),
      });
      const data = await res.json();
      if (data.success && data.workflow) {
        onImport(data.workflow);
        onClose();
      } else {
        alert('Import failed: ' + (data.error || 'Unknown error'));
      }
    } catch (err) {
      alert('Import failed: ' + err.message);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content import-modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Import from AI Scientist</h2>
          <button className="close-btn" onClick={onClose}>&times;</button>
        </div>
        
        {loading && <div className="loading-state">Fetching survivors...</div>}
        {error && <div className="error-state">{error}</div>}
        
        <div className="survivor-list">
          {!loading && survivors.length === 0 && <p className="muted">No survivors found in the notebook.</p>}
          {survivors.map(s => (
            <div key={s.result_id} className="survivor-item">
              <div className="survivor-info">
                <strong>{s.name || s.result_id}</strong>
                <div className="survivor-meta">
                  Loss Ratio: {s.loss_ratio?.toFixed(4)} | Novelty: {s.novelty_score?.toFixed(2)}
                </div>
              </div>
              <button className="primary small" onClick={() => handleImport(s.result_id)}>Import</button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ImportDialog;
