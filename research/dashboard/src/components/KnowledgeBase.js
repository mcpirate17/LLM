import React, { useState, useEffect } from 'react';

const API_BASE = process.env.REACT_APP_API_URL || '';

const CATEGORY_COLORS = {
  principle: 'var(--accent-blue)',
  anti_pattern: 'var(--accent-red)',
  sweet_spot: 'var(--accent-green)',
  correlation: 'var(--accent-purple)',
  tool_insight: 'var(--accent-yellow)',
};

const CATEGORY_LABELS = {
  principle: 'Principle',
  anti_pattern: 'Anti-Pattern',
  sweet_spot: 'Sweet Spot',
  correlation: 'Correlation',
  tool_insight: 'Tool Insight',
};

function ConfidenceBar({ confidence }) {
  const pct = Math.min(confidence * 100, 100);
  const color = confidence >= 0.7 ? 'var(--accent-green)' :
                confidence >= 0.4 ? 'var(--accent-yellow)' : 'var(--accent-red)';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <div style={{ flex: 1, height: 6, background: 'var(--bg-tertiary)', borderRadius: 3, maxWidth: 80 }}>
        <div style={{
          width: `${pct}%`, height: '100%',
          background: color, borderRadius: 3,
        }} />
      </div>
      <span style={{ fontSize: 11, color: 'var(--text-muted)', minWidth: 30 }}>{pct.toFixed(0)}%</span>
    </div>
  );
}

function KnowledgeBase() {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState(null);
  const [search, setSearch] = useState('');
  const [expanded, setExpanded] = useState({});

  useEffect(() => {
    const url = filter
      ? `${API_BASE}/api/knowledge?category=${filter}`
      : `${API_BASE}/api/knowledge`;
    fetch(url)
      .then(r => r.json())
      .then(d => { setEntries(Array.isArray(d) ? d : []); setLoading(false); })
      .catch(() => setLoading(false));
  }, [filter]);

  const doSearch = async () => {
    if (!search.trim()) return;
    setLoading(true);
    try {
      const r = await fetch(`${API_BASE}/api/knowledge/search?q=${encodeURIComponent(search)}`);
      const d = await r.json();
      setEntries(Array.isArray(d) ? d : []);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const toggleExpand = (id) => {
    setExpanded(prev => ({ ...prev, [id]: !prev[id] }));
  };

  const categories = ['principle', 'anti_pattern', 'sweet_spot', 'correlation', 'tool_insight'];

  return (
    <div>
      <h2 style={{ fontSize: 16, marginBottom: 16 }}>Knowledge Base</h2>

      {/* Filters */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap', alignItems: 'center' }}>
        <button
          className={`tab ${filter === null ? 'active' : ''}`}
          onClick={() => setFilter(null)}
          style={{ padding: '4px 12px', fontSize: 12 }}
        >
          All
        </button>
        {categories.map(cat => (
          <button
            key={cat}
            className={`tab ${filter === cat ? 'active' : ''}`}
            onClick={() => setFilter(cat)}
            style={{
              padding: '4px 12px', fontSize: 12,
              color: filter === cat ? CATEGORY_COLORS[cat] : undefined,
            }}
          >
            {CATEGORY_LABELS[cat] || cat}
          </button>
        ))}

        <div style={{ marginLeft: 'auto', display: 'flex', gap: 4 }}>
          <input
            type="text"
            placeholder="Search..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && doSearch()}
            style={{ padding: '4px 8px', fontSize: 12, width: 160 }}
          />
          <button className="refresh-btn" onClick={doSearch} style={{ padding: '4px 8px', fontSize: 12 }}>
            Search
          </button>
        </div>
      </div>

      {/* Entries */}
      {loading ? (
        <p style={{ color: 'var(--text-muted)' }}>Loading...</p>
      ) : entries.length === 0 ? (
        <p style={{ color: 'var(--text-muted)' }}>
          No knowledge entries yet. Knowledge is extracted automatically during continuous experiments.
        </p>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {entries.map(entry => (
            <div
              key={entry.entry_id}
              className="card"
              style={{
                padding: 12, cursor: 'pointer',
                borderLeft: `3px solid ${CATEGORY_COLORS[entry.category] || 'var(--border)'}`,
              }}
              onClick={() => toggleExpand(entry.entry_id)}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{
                    padding: '2px 6px', borderRadius: 3, fontSize: 10,
                    fontWeight: 600, textTransform: 'uppercase',
                    color: CATEGORY_COLORS[entry.category] || 'var(--text-muted)',
                    background: `${CATEGORY_COLORS[entry.category] || 'var(--text-muted)'}22`,
                  }}>
                    {CATEGORY_LABELS[entry.category] || entry.category}
                  </span>
                  <span style={{ fontSize: 13, fontWeight: 500 }}>{entry.title}</span>
                  {entry.times_validated > 1 && (
                    <span style={{
                      padding: '1px 5px', borderRadius: 8, fontSize: 10,
                      background: 'var(--accent-green)22', color: 'var(--accent-green)',
                      fontWeight: 600,
                    }}>
                      {entry.times_validated}x validated
                    </span>
                  )}
                </div>
                <ConfidenceBar confidence={entry.confidence || 0} />
              </div>

              <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 6 }}>
                {entry.content}
              </div>

              {expanded[entry.entry_id] && entry.supporting_evidence && (
                <div style={{
                  marginTop: 8, padding: 8,
                  background: 'var(--bg-tertiary)', borderRadius: 4,
                  fontSize: 11, color: 'var(--text-muted)',
                }}>
                  <strong>Supporting Evidence:</strong>{' '}
                  {Array.isArray(entry.supporting_evidence)
                    ? entry.supporting_evidence.join(', ')
                    : String(entry.supporting_evidence)}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default KnowledgeBase;
