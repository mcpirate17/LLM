import React, { useState, useEffect, useMemo } from 'react';
import { confidenceColor } from '../utils/colors';
import useCopyToClipboard from '../hooks/useCopyToClipboard';

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

const KNOWLEDGE_CATEGORIES = ['principle', 'anti_pattern', 'sweet_spot', 'correlation', 'tool_insight'];
const KNOWLEDGE_BASE_PREFS_KEY = 'dashboard.knowledge-base.prefs.v1';

function ConfidenceBar({ confidence }) {
  const pct = Math.min(confidence * 100, 100);
  const color = confidenceColor(confidence);
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

function KnowledgeBase({ onSelectExperiment }) {
  const [allEntries, setAllEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchLoading, setSearchLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(KNOWLEDGE_BASE_PREFS_KEY) || '{}');
      if (stored.filter === null) return null;
      if (typeof stored.filter === 'string' && KNOWLEDGE_CATEGORIES.includes(stored.filter)) {
        return stored.filter;
      }
    } catch {}
    return null;
  });
  const [search, setSearch] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(KNOWLEDGE_BASE_PREFS_KEY) || '{}');
      if (typeof stored.search === 'string') {
        return stored.search;
      }
    } catch {}
    return '';
  });
  const [expanded, setExpanded] = useState({});
  const [lastUpdated, setLastUpdated] = useState(null);
  const [copiedValue, copyText] = useCopyToClipboard();
  const [isSearchResult, setIsSearchResult] = useState(false);

  useEffect(() => {
    setError(null);
    fetch(`${API_BASE}/api/knowledge`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(d => { setAllEntries(Array.isArray(d) ? d : []); setIsSearchResult(false); setLastUpdated(new Date()); setLoading(false); })
      .catch(e => { setError('Failed to load knowledge base: ' + e.message); setLoading(false); });
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(KNOWLEDGE_BASE_PREFS_KEY, JSON.stringify({ filter, search }));
    } catch {}
  }, [filter, search]);

  const categoryCounts = useMemo(() => {
    const counts = {};
    for (const e of allEntries) {
      const cat = e.category || 'unknown';
      counts[cat] = (counts[cat] || 0) + 1;
    }
    return counts;
  }, [allEntries]);

  const entries = useMemo(() => {
    if (isSearchResult) return allEntries;
    if (!filter) return allEntries;
    return allEntries.filter(e => e.category === filter);
  }, [allEntries, filter, isSearchResult]);

  const doSearch = async () => {
    if (!search.trim()) {
      // Clear search — reload all
      setIsSearchResult(false);
      setFilter(null);
      return;
    }
    setSearchLoading(true);
    setError(null);
    try {
      const r = await fetch(`${API_BASE}/api/knowledge/search?q=${encodeURIComponent(search)}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      setAllEntries(Array.isArray(d) ? d : []);
      setIsSearchResult(true);
      setFilter(null);
      setLastUpdated(new Date());
    } catch (e) {
      setError('Search failed: ' + e.message);
    }
    setSearchLoading(false);
  };

  const extractExperimentIds = (supportingEvidence) => {
    const raw = Array.isArray(supportingEvidence)
      ? supportingEvidence.map(String)
      : [String(supportingEvidence || '')];
    const ids = raw
      .flatMap((item) => item.split(/[\s,;]+/))
      .map((item) => item.trim())
      .filter((item) => /^exp[_-][A-Za-z0-9_-]+$/.test(item));
    return Array.from(new Set(ids));
  };

  const toggleExpand = (id) => {
    setExpanded(prev => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <div>
      <h2 style={{ fontSize: 16, marginBottom: 16 }}>Knowledge Base</h2>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Curated lessons extracted from past experiments, such as reliable design principles and common failure
        patterns. Confidence shows how strongly the existing evidence supports each claim.
      </p>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10 }}>
        Last updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : 'loading'} · Source: /api/knowledge
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap', alignItems: 'center' }}>
        <button
          className={`tab ${filter === null ? 'active' : ''}`}
          onClick={() => { setFilter(null); setIsSearchResult(false); }}
          aria-label="Show all knowledge categories"
          style={{ padding: '4px 12px', fontSize: 12 }}
        >
          All ({allEntries.length})
        </button>
        {KNOWLEDGE_CATEGORIES.map(cat => (
          <button
            key={cat}
            className={`tab ${filter === cat ? 'active' : ''}`}
            onClick={() => { setFilter(cat); setIsSearchResult(false); }}
            aria-label={`Filter knowledge by ${CATEGORY_LABELS[cat] || cat}`}
            style={{
              padding: '4px 12px', fontSize: 12,
              color: filter === cat ? CATEGORY_COLORS[cat] : undefined,
            }}
          >
            {CATEGORY_LABELS[cat] || cat}{categoryCounts[cat] ? ` (${categoryCounts[cat]})` : ''}
          </button>
        ))}

        <div style={{ marginLeft: 'auto', display: 'flex', gap: 4 }}>
          <input
            type="text"
            placeholder="Search..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && doSearch()}
            aria-label="Search knowledge base"
            style={{ padding: '4px 8px', fontSize: 12, width: 160 }}
          />
          <button className="refresh-btn" onClick={doSearch} aria-label="Run knowledge base search" style={{ padding: '4px 8px', fontSize: 12 }}>
            {searchLoading ? 'Searching...' : 'Search'}
          </button>
        </div>
      </div>

      {searchLoading && (
        <p style={{ color: 'var(--text-muted)', marginTop: -6, marginBottom: 10, fontSize: 12 }}>
          Searching knowledge base...
        </p>
      )}

      {/* Entries */}
      {error && (
        <p style={{ color: 'var(--accent-red)', marginBottom: 8 }}>{error}</p>
      )}
      {loading ? (
        <p style={{ color: 'var(--text-muted)' }}>Loading...</p>
      ) : entries.length === 0 && !error ? (
        <div style={{ color: 'var(--text-muted)', fontSize: 13, lineHeight: 1.6 }}>
          <p style={{ margin: 0 }}>
            No knowledge entries found.
          </p>
          <p style={{ margin: '6px 0 0' }}>
            Run continuous experiments to accumulate evidence, or clear filters/search to view all available insights.
          </p>
        </div>
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
                  <span
                    style={{
                      fontSize: 13,
                      fontWeight: 500,
                      maxWidth: 360,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                    title={entry.title || 'not available'}
                  >
                    {entry.title}
                  </span>
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

              <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 6, whiteSpace: expanded[entry.entry_id] ? 'pre-wrap' : 'normal' }}
                title={!expanded[entry.entry_id] ? (entry.content || 'not available') : undefined}>
                {expanded[entry.entry_id]
                  ? entry.content
                  : (entry.content || '').split('\n')[0]?.slice(0, 150) + ((entry.content || '').length > 150 || (entry.content || '').includes('\n') ? '...' : '')
                }
              </div>

              <div style={{ marginTop: 8 }}>
                <button
                  className="refresh-btn"
                  style={{ fontSize: 10, padding: '2px 6px' }}
                  onClick={(e) => {
                    e.stopPropagation();
                    copyText(entry.entry_id);
                  }}
                  aria-label={`Copy knowledge entry id ${entry.entry_id}`}
                >
                  {copiedValue === entry.entry_id ? 'Copied Entry ID' : 'Copy Entry ID'}
                </button>
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
                  {extractExperimentIds(entry.supporting_evidence).length > 0 && (
                    <div style={{ marginTop: 8, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                      {extractExperimentIds(entry.supporting_evidence).map((expId) => (
                        <button
                          key={expId}
                          className="refresh-btn"
                          style={{ fontSize: 10, padding: '2px 6px' }}
                          onClick={(e) => {
                            e.stopPropagation();
                            if (onSelectExperiment) onSelectExperiment(expId);
                          }}
                          aria-label={`Open experiment ${expId} from supporting evidence`}
                          disabled={!onSelectExperiment}
                        >
                          Open {expId.slice(0, 12)}
                        </button>
                      ))}
                    </div>
                  )}
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
