import { apiCall } from "../services/apiService";
import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { confidenceColor } from '../utils/colors';
import useCopyToClipboard from '../hooks/useCopyToClipboard';


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
const STOPWORDS = new Set([
  'the', 'and', 'for', 'that', 'with', 'this', 'from', 'into', 'when', 'then', 'than', 'were', 'been',
  'have', 'has', 'had', 'are', 'was', 'show', 'shows', 'showed', 'over', 'under', 'across', 'between',
  'using', 'use', 'used', 'high', 'low', 'very', 'more', 'less', 'near', 'around', 'recent', 'experiments',
  'experiment', 'result', 'results', 'indicate', 'indicates', 'suggest', 'suggests', 'mode', 'patterns',
  'pattern', 'architecture', 'architectures',
]);

function canonicalize(raw) {
  return String(raw || '')
    .toLowerCase()
    .replace(/\b\d+(\.\d+)?%?\b/g, '#')
    .replace(/[^a-z0-9#\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function entrySignalScore(entry) {
  const effective = Number(entry.effective_confidence ?? entry.confidence ?? 0.5);
  const validations = Math.max(0, Number(entry.times_validated || 0));
  const validationBoost = Math.min(0.12, Math.log1p(validations) * 0.03);
  return Math.max(0, effective + validationBoost);
}

function isLowSignalEntry(entry) {
  const title = canonicalize(entry.title);
  const content = canonicalize(entry.content);
  if (!title || !content) return true;
  if (title.length < 16 || content.length < 60) return true;
  if (title.startsWith('recent experiments show') || title.startsWith('all recent experiments show')) return true;
  if (String(entry.content || '').includes('...') || String(entry.title || '').includes('...')) return true;
  if (/\$|\\approx/.test(String(entry.content || ''))) return true;
  return false;
}

function clusterKeyForEntry(entry) {
  const category = entry.category || 'unknown';
  const tokens = `${entry.title || ''} ${String(entry.content || '').split('\n')[0] || ''}`
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((token) => token && token.length > 3 && !STOPWORDS.has(token))
    .slice(0, 6);
  const stem = tokens.slice(0, 4).join('_') || canonicalize(entry.title).split(' ').slice(0, 4).join('_') || 'misc';
  return `${category}:${stem}`;
}

function formatPct(v) {
  return `${Math.round((Number(v) || 0) * 100)}%`;
}

function entryTokens(entry) {
  return new Set(
    `${entry.title || ''} ${entry.content || ''}`
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter((token) => token && token.length > 3 && !STOPWORDS.has(token))
  );
}

function jaccard(a, b) {
  if (!a.size || !b.size) return 0;
  let inter = 0;
  const seen = new Set([...a, ...b]);
  for (const tok of seen) {
    if (a.has(tok) && b.has(tok)) inter += 1;
  }
  return inter / seen.size;
}

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
  const [expandedClusters, setExpandedClusters] = useState({});
  const [clusterVisibleCount, setClusterVisibleCount] = useState({});
  const [showLowSignal, setShowLowSignal] = useState(false);
  const [showSingletonClusters, setShowSingletonClusters] = useState(false);
  const [visibleClusterCount, setVisibleClusterCount] = useState(8);
  const [expandedCategories, setExpandedCategories] = useState({});
  const [lastUpdated, setLastUpdated] = useState(null);
  const [copiedValue, copyText] = useCopyToClipboard();
  const [isSearchResult, setIsSearchResult] = useState(false);

  const fetchAllEntries = useCallback(async () => {
    setError(null);
    const r = await apiCall(`/api/knowledge`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    setAllEntries(Array.isArray(d) ? d : []);
    setIsSearchResult(false);
    setLastUpdated(new Date());
  }, []);

  useEffect(() => {
    fetchAllEntries()
      .catch(e => setError('Failed to load knowledge base: ' + e.message))
      .finally(() => setLoading(false));
  }, [fetchAllEntries]);

  useEffect(() => {
    setExpandedClusters({});
    setClusterVisibleCount({});
    setVisibleClusterCount(8);
  }, [filter, search, isSearchResult, showLowSignal, showSingletonClusters]);

  useEffect(() => {
    setExpandedClusters({});
    setClusterVisibleCount({});
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
    const base = isSearchResult
      ? allEntries
      : (!filter ? allEntries : allEntries.filter(e => e.category === filter));
    if (showLowSignal) return base;
    return base.filter((entry) => !isLowSignalEntry(entry) && entrySignalScore(entry) >= 0.6);
  }, [allEntries, filter, isSearchResult, showLowSignal]);

  const allClusters = useMemo(() => {
    const semanticClusters = [];
    for (const entry of entries) {
      const category = entry.category || 'unknown';
      const tokens = entryTokens(entry);
      const score = entrySignalScore(entry);
      const validations = Number(entry.times_validated || 0);
      const effective = Number(entry.effective_confidence ?? entry.confidence ?? 0.5);
      let cluster = semanticClusters.find((c) => {
        if (c.category !== category) return false;
        const sim = jaccard(tokens, c.tokenSet);
        const overlap = [...tokens].filter((tok) => c.tokenSet.has(tok)).length;
        return sim >= 0.16 && overlap >= 4;
      });
      if (!cluster) {
        cluster = {
          key: clusterKeyForEntry(entry),
          category,
          entries: [],
          scoreSum: 0,
          totalValidations: 0,
          maxConfidence: 0,
          newestTs: 0,
          representative: entry,
          tokenSet: new Set(tokens),
        };
        semanticClusters.push(cluster);
      }
      for (const tok of tokens) cluster.tokenSet.add(tok);
      cluster.entries.push(entry);
      cluster.scoreSum += score;
      cluster.totalValidations += validations;
      cluster.maxConfidence = Math.max(cluster.maxConfidence, effective);
      cluster.newestTs = Math.max(cluster.newestTs, Number(entry.timestamp || 0));
      const repScore = entrySignalScore(cluster.representative);
      if (score > repScore || (score === repScore && Number(entry.timestamp || 0) > Number(cluster.representative.timestamp || 0))) {
        cluster.representative = entry;
      }
    }
    const out = semanticClusters.map((cluster) => {
      const avgScore = cluster.entries.length ? cluster.scoreSum / cluster.entries.length : 0;
      const validatedBoost = Math.min(0.14, Math.log1p(cluster.totalValidations) * 0.03);
      const sizeBoost = Math.min(0.08, Math.log1p(cluster.entries.length) * 0.04);
      const clusterScore = avgScore + validatedBoost + sizeBoost;
      const sortedEntries = cluster.entries
        .slice()
        .sort((a, b) => {
          const diff = entrySignalScore(b) - entrySignalScore(a);
          if (diff !== 0) return diff;
          return Number(b.timestamp || 0) - Number(a.timestamp || 0);
        });
      return {
        ...cluster,
        entries: sortedEntries,
        avgScore,
        clusterScore,
      };
    });
    out.sort((a, b) => b.clusterScore - a.clusterScore);
    return out;
  }, [entries]);

  const clusters = useMemo(
    () => allClusters.filter((cluster) => showSingletonClusters || cluster.entries.length > 1),
    [allClusters, showSingletonClusters]
  );

  const hiddenSingletonCount = useMemo(() => {
    if (showSingletonClusters) return 0;
    return allClusters.filter((cluster) => cluster.entries.length <= 1).length;
  }, [allClusters, showSingletonClusters]);

  const compactDigest = useMemo(() => {
    const lines = [];
    for (const cluster of clusters.slice(0, 12)) {
      const rep = cluster.representative;
      lines.push(
        `[${CATEGORY_LABELS[cluster.category] || cluster.category}] ` +
        `${rep.title || 'Untitled'} | signal=${formatPct(cluster.avgScore)} ` +
        `| entries=${cluster.entries.length} | validations=${cluster.totalValidations}`
      );
    }
    return lines.join('\n');
  }, [clusters]);

  const totalSuppressed = useMemo(() => {
    const base = isSearchResult
      ? allEntries
      : (!filter ? allEntries : allEntries.filter(e => e.category === filter));
    return Math.max(0, base.length - entries.length);
  }, [allEntries, entries, filter, isSearchResult]);

  const doSearch = async () => {
    const q = search.trim();
    if (!q) {
      setFilter(null);
      setSearch('');
      setSearchLoading(true);
      try {
        await fetchAllEntries();
      } catch (e) {
        setError('Failed to load knowledge base: ' + e.message);
      }
      setSearchLoading(false);
      return;
    }
    setSearchLoading(true);
    setError(null);
    try {
      const r = await apiCall(`/api/knowledge/search?q=${encodeURIComponent(search)}`);
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

  const toggleClusterExpand = (id) => {
    setExpandedClusters((prev) => ({ ...prev, [id]: !prev[id] }));
    setClusterVisibleCount((prev) => ({ ...prev, [id]: prev[id] || 3 }));
  };

  const showMoreClusterEntries = (id, total) => {
    setClusterVisibleCount((prev) => {
      const next = Math.min(total, (prev[id] || 3) + 5);
      return { ...prev, [id]: next };
    });
  };

  const topClusterCount = clusters.length;
  const topInsightCount = entries.length;
  const visibleClusters = clusters.slice(0, visibleClusterCount);
  const clustersByCategory = useMemo(() => {
    const grouped = new Map();
    for (const cluster of visibleClusters) {
      const key = cluster.category || 'unknown';
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key).push(cluster);
    }
    return Array.from(grouped.entries())
      .map(([category, items]) => ({
        category,
        items,
        insights: items.reduce((acc, c) => acc + c.entries.length, 0),
        avgSignal: items.length ? (items.reduce((acc, c) => acc + c.avgScore, 0) / items.length) : 0,
      }))
      .sort((a, b) => b.avgSignal - a.avgSignal);
  }, [visibleClusters]);

  useEffect(() => {
    if (!clustersByCategory.length) {
      setExpandedCategories({});
      return;
    }
    setExpandedCategories((prev) => {
      const next = {};
      clustersByCategory.forEach((group, idx) => {
        next[group.category] = Object.prototype.hasOwnProperty.call(prev, group.category)
          ? !!prev[group.category]
          : idx < 2;
      });
      return next;
    });
  }, [clustersByCategory]);

  const toggleCategory = (category) => {
    setExpandedCategories((prev) => ({ ...prev, [category]: !prev[category] }));
  };

  return (
    <div>
      <h2 style={{ fontSize: 16, marginBottom: 16 }}>Knowledge Base</h2>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Curated lessons extracted from past experiments, such as reliable design principles and common failure
        patterns. Insights are clustered by theme and ranked by validation-weighted confidence.
      </p>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10 }}>
        Last updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : 'loading'} · Source: /api/knowledge
      </div>

      <div className="knowledge-summary-row">
        <div className="knowledge-summary-stat">
          <strong>{topClusterCount}</strong>
          <span>clusters</span>
        </div>
        <div className="knowledge-summary-stat">
          <strong>{topInsightCount}</strong>
          <span>insights shown</span>
        </div>
        {totalSuppressed > 0 && (
          <div className="knowledge-summary-stat muted">
            <strong>{totalSuppressed}</strong>
            <span>low-signal hidden</span>
          </div>
        )}
        {!showSingletonClusters && hiddenSingletonCount > 0 && (
          <div className="knowledge-summary-stat muted">
            <strong>{hiddenSingletonCount}</strong>
            <span>singletons hidden</span>
          </div>
        )}
        <button
          className="refresh-btn"
          style={{ marginLeft: 'auto', padding: '4px 8px', fontSize: 12 }}
          onClick={() => copyText(compactDigest)}
          disabled={!compactDigest}
          aria-label="Copy compact insight digest"
        >
          {copiedValue === compactDigest ? 'Copied Digest' : 'Copy Compact Digest'}
        </button>
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

        <button
          className={`refresh-btn ${showLowSignal ? 'active' : ''}`}
          style={{ padding: '4px 10px', fontSize: 12 }}
          onClick={() => setShowLowSignal((v) => !v)}
          aria-pressed={showLowSignal}
          aria-label="Toggle low signal insights"
        >
          {showLowSignal ? 'Including low-signal' : 'High-signal only'}
        </button>

        <button
          className={`refresh-btn ${showSingletonClusters ? 'active' : ''}`}
          style={{ padding: '4px 10px', fontSize: 12 }}
          onClick={() => setShowSingletonClusters((v) => !v)}
          aria-pressed={showSingletonClusters}
          aria-label="Toggle singleton clusters"
        >
          {showSingletonClusters ? 'Showing singletons' : 'Hide singletons'}
        </button>

        <div style={{ marginLeft: 'auto', display: 'flex', gap: 4 }}>
          <input
            type="text"
            placeholder="Search..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && doSearch()}
            aria-label="Search knowledge base"
            style={{ padding: '4px 8px', fontSize: 12, width: 220 }}
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
      ) : clusters.length === 0 && !error ? (
        <div style={{ color: 'var(--text-muted)', fontSize: 13, lineHeight: 1.6 }}>
          <p style={{ margin: 0 }}>
            No knowledge entries found.
          </p>
          <p style={{ margin: '6px 0 0' }}>
            Run continuous experiments to accumulate evidence, or clear filters/search to view all available insights.
          </p>
        </div>
      ) : (
        <div className="knowledge-cluster-list">
          {clustersByCategory.map((group) => (
            <div key={group.category} className="knowledge-category-section">
              <button
                className="knowledge-category-header"
                onClick={() => toggleCategory(group.category)}
                aria-expanded={!!expandedCategories[group.category]}
              >
                <span className="knowledge-category-title">
                  {CATEGORY_LABELS[group.category] || group.category}
                </span>
                <span className="knowledge-category-metrics">
                  {group.items.length} clusters · {group.insights} insights · {formatPct(group.avgSignal)} signal
                </span>
              </button>

              {!!expandedCategories[group.category] && group.items.map((cluster) => {
            const isOpen = !!expandedClusters[cluster.key];
            const visible = clusterVisibleCount[cluster.key] || 3;
            const shownEntries = cluster.entries.slice(0, visible);
            const rep = cluster.representative;
            const categoryColor = CATEGORY_COLORS[cluster.category] || 'var(--border)';
            return (
              <div
                key={cluster.key}
                className="card knowledge-cluster-card"
                style={{ borderLeft: `3px solid ${categoryColor}` }}
              >
                <button
                  className="knowledge-cluster-header"
                  onClick={() => toggleClusterExpand(cluster.key)}
                  aria-expanded={isOpen}
                  aria-label={`Toggle cluster ${rep.title || 'untitled insight cluster'}`}
                >
                  <div className="knowledge-cluster-title-wrap">
                    <span
                      className="knowledge-cluster-category"
                      style={{
                        color: categoryColor,
                        background: `${categoryColor}22`,
                      }}
                    >
                      {CATEGORY_LABELS[cluster.category] || cluster.category}
                    </span>
                    <span className="knowledge-cluster-title" title={rep.title || 'not available'}>
                      {rep.title}
                    </span>
                  </div>
                  <div className="knowledge-cluster-metrics">
                    <span>{cluster.entries.length} insights</span>
                    <span>{cluster.totalValidations} validations</span>
                    <span>{formatPct(cluster.avgScore)} signal</span>
                  </div>
                </button>

                <div className="knowledge-cluster-summary" title={rep.content || ''}>
                  {rep.content}
                </div>

                {isOpen && (
                  <div className="knowledge-cluster-entries">
                    {shownEntries.map((entry) => (
                      <div
                        key={entry.entry_id}
                        className="knowledge-entry-row"
                        onClick={() => toggleExpand(entry.entry_id)}
                      >
                        <div className="knowledge-entry-row-header">
                          <span className="knowledge-entry-title" title={entry.title || 'not available'}>
                            {entry.title}
                          </span>
                          <ConfidenceBar confidence={entry.effective_confidence ?? entry.confidence ?? 0} />
                        </div>
                        <div
                          className="knowledge-entry-content"
                          style={{ whiteSpace: expanded[entry.entry_id] ? 'pre-wrap' : 'normal' }}
                          title={!expanded[entry.entry_id] ? (entry.content || 'not available') : undefined}
                        >
                          {expanded[entry.entry_id]
                            ? entry.content
                            : (entry.content || '').split('\n')[0]?.slice(0, 175) + ((entry.content || '').length > 175 || (entry.content || '').includes('\n') ? '...' : '')
                          }
                        </div>

                        <div className="knowledge-entry-actions">
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
                          {Number(entry.times_validated || 0) > 1 && (
                            <span className="knowledge-entry-badge">
                              {entry.times_validated}x validated
                            </span>
                          )}
                        </div>

                        {expanded[entry.entry_id] && entry.supporting_evidence && (
                          <div className="knowledge-entry-evidence">
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

                    {cluster.entries.length > shownEntries.length && (
                      <div style={{ marginTop: 6 }}>
                        <button
                          className="refresh-btn"
                          style={{ fontSize: 11, padding: '3px 8px' }}
                          onClick={(e) => {
                            e.stopPropagation();
                            showMoreClusterEntries(cluster.key, cluster.entries.length);
                          }}
                        >
                          Show more ({cluster.entries.length - shownEntries.length} remaining)
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
            </div>
          ))}
          {clusters.length > visibleClusters.length && (
            <div className="knowledge-load-more-wrap">
              <button
                className="refresh-btn"
                style={{ padding: '6px 12px', fontSize: 12 }}
                onClick={() => setVisibleClusterCount((n) => Math.min(clusters.length, n + 8))}
              >
                Load More Clusters ({clusters.length - visibleClusters.length} remaining)
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default KnowledgeBase;
