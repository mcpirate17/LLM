import React, { useState, useEffect, useCallback, useRef } from 'react';
import apiService from '../services/apiService';

// Color mapping for op types
const OP_COLORS = {
  // Unary ops
  relu: '#58a6ff',
  gelu: '#58a6ff',
  silu: '#58a6ff',
  sigmoid: '#58a6ff',
  tanh: '#58a6ff',
  softmax: '#58a6ff',
  layer_norm: '#58a6ff',
  rms_norm: '#58a6ff',
  dropout: '#58a6ff',
  // Binary ops
  add: '#3fb950',
  mul: '#3fb950',
  sub: '#3fb950',
  cat: '#3fb950',
  // MatMul / linear
  matmul: '#f85149',
  linear: '#f85149',
  conv1d: '#f85149',
  conv2d: '#f85149',
  // Attention
  attention: '#bc8cff',
  multi_head_attention: '#bc8cff',
  // Embedding / positional
  embedding: '#d29922',
  pos_encoding: '#d29922',
  // Default
  _default: '#8b949e',
};

function getOpColor(opName) {
  const key = (opName || '').toLowerCase();
  return OP_COLORS[key] || OP_COLORS._default;
}

function getOpCategory(opName) {
  const key = (opName || '').toLowerCase();
  if (['relu', 'gelu', 'silu', 'sigmoid', 'tanh', 'softmax', 'layer_norm', 'rms_norm', 'dropout'].includes(key)) return 'unary';
  if (['add', 'mul', 'sub', 'cat'].includes(key)) return 'binary';
  if (['matmul', 'linear', 'conv1d', 'conv2d'].includes(key)) return 'matmul';
  if (['attention', 'multi_head_attention'].includes(key)) return 'attention';
  if (['embedding', 'pos_encoding'].includes(key)) return 'embedding';
  return 'other';
}

function formatMemory(bytes) {
  if (bytes == null) return '--';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatDuration(us) {
  if (us == null) return '--';
  if (us < 1000) return `${us.toFixed(1)} us`;
  if (us < 1000000) return `${(us / 1000).toFixed(2)} ms`;
  return `${(us / 1000000).toFixed(3)} s`;
}

function NativeProfilePanel() {
  const [profile, setProfile] = useState(null);
  const [enabled, setEnabled] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [toggling, setToggling] = useState(false);
  const [expanded, setExpanded] = useState(true);
  const [hoveredNode, setHoveredNode] = useState(null);
  const intervalRef = useRef(null);

  const fetchProfile = useCallback(async () => {
    try {
      const data = await apiService.getNativeProfile();
      setProfile(data);
      setEnabled(!!data.enabled);
      setError(null);
    } catch (err) {
      setError(err.message || 'Failed to fetch profile data');
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchProfile();
  }, [fetchProfile]);

  // Auto-refresh when enabled
  useEffect(() => {
    if (enabled) {
      intervalRef.current = setInterval(fetchProfile, 2000);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [enabled, fetchProfile]);

  const handleToggle = useCallback(async () => {
    setToggling(true);
    try {
      await apiService.toggleNativeProfiling(!enabled);
      setEnabled(!enabled);
      if (!enabled) {
        // Switching to enabled -- fetch fresh data shortly
        setTimeout(fetchProfile, 500);
      }
    } catch (err) {
      setError(err.message || 'Failed to toggle profiling');
    } finally {
      setToggling(false);
    }
  }, [enabled, fetchProfile]);

  const nodes = profile?.node_profiles || [];
  const maxDuration = nodes.length > 0 ? Math.max(...nodes.map(n => n.duration_us)) : 0;
  const totalDuration = profile?.total_duration_us;
  const peakMemory = profile?.peak_memory_bytes;

  // Legend categories present in data
  const categoriesPresent = [...new Set(nodes.map(n => getOpCategory(n.op_name)))];
  const categoryLabels = {
    unary: 'Unary',
    binary: 'Binary',
    matmul: 'MatMul',
    attention: 'Attention',
    embedding: 'Embedding',
    other: 'Other',
  };
  const categoryColors = {
    unary: '#58a6ff',
    binary: '#3fb950',
    matmul: '#f85149',
    attention: '#bc8cff',
    embedding: '#d29922',
    other: '#8b949e',
  };

  const sectionStyle = {
    background: 'var(--bg-secondary)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    marginBottom: 16,
    overflow: 'hidden',
  };

  const headerStyle = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '10px 14px',
    cursor: 'pointer',
    userSelect: 'none',
  };

  const badgeStyle = (color) => ({
    display: 'inline-block',
    padding: '2px 8px',
    borderRadius: 12,
    fontSize: 11,
    fontWeight: 600,
    background: `${color}20`,
    color,
    border: `1px solid ${color}40`,
  });

  return (
    <div style={sectionStyle}>
      {/* Header / Toggle */}
      <div style={headerStyle} onClick={() => setExpanded(!expanded)}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'monospace' }}>
            {expanded ? '\u25BC' : '\u25B6'}
          </span>
          <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>
            Native Runner Profiler
          </span>
          <span style={badgeStyle(enabled ? 'var(--accent-green)' : 'var(--text-muted)')}>
            {enabled ? 'Active' : 'Disabled'}
          </span>
          {nodes.length > 0 && (
            <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
              {nodes.length} nodes &middot; {formatDuration(totalDuration)}
            </span>
          )}
        </div>
        <button
          className="refresh-btn"
          style={{ fontSize: 11, padding: '3px 10px' }}
          onClick={(e) => { e.stopPropagation(); handleToggle(); }}
          disabled={toggling}
        >
          {toggling ? '...' : (enabled ? 'Disable Profiling' : 'Enable Profiling')}
        </button>
      </div>

      {expanded && (
        <div style={{ padding: '0 14px 14px' }}>
          {error && (
            <div style={{
              padding: '8px 12px',
              marginBottom: 12,
              background: 'rgba(248, 81, 73, 0.1)',
              border: '1px solid var(--accent-red)',
              borderRadius: 'var(--radius)',
              color: 'var(--accent-red)',
              fontSize: 12,
            }}>
              {error}
            </div>
          )}

          {!enabled && nodes.length === 0 && (
            <div style={{
              textAlign: 'center',
              padding: '24px 16px',
              color: 'var(--text-muted)',
              fontSize: 13,
            }}>
              Profiling is disabled. Enable it to collect per-node timing data.
            </div>
          )}

          {enabled && nodes.length === 0 && (
            <div style={{
              textAlign: 'center',
              padding: '24px 16px',
              color: 'var(--text-muted)',
              fontSize: 13,
            }}>
              Profiling enabled &mdash; waiting for data. Run an experiment to collect timing data.
            </div>
          )}

          {nodes.length > 0 && (
            <>
              {/* Summary stats */}
              <div style={{
                display: 'flex',
                gap: 16,
                marginBottom: 14,
                flexWrap: 'wrap',
              }}>
                <div style={{
                  flex: '1 1 120px',
                  background: 'var(--bg-tertiary)',
                  borderRadius: 'var(--radius)',
                  padding: '10px 14px',
                }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Total Duration</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--accent-blue)', fontFamily: 'monospace' }}>
                    {formatDuration(totalDuration)}
                  </div>
                </div>
                <div style={{
                  flex: '1 1 120px',
                  background: 'var(--bg-tertiary)',
                  borderRadius: 'var(--radius)',
                  padding: '10px 14px',
                }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Peak Memory</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--accent-yellow)', fontFamily: 'monospace' }}>
                    {formatMemory(peakMemory)}
                  </div>
                </div>
                <div style={{
                  flex: '1 1 120px',
                  background: 'var(--bg-tertiary)',
                  borderRadius: 'var(--radius)',
                  padding: '10px 14px',
                }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Node Count</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--accent-green)', fontFamily: 'monospace' }}>
                    {nodes.length}
                  </div>
                </div>
              </div>

              {/* Legend */}
              {categoriesPresent.length > 1 && (
                <div style={{
                  display: 'flex',
                  gap: 12,
                  marginBottom: 10,
                  flexWrap: 'wrap',
                  fontSize: 11,
                  color: 'var(--text-secondary)',
                }}>
                  {categoriesPresent.map(cat => (
                    <div key={cat} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <span style={{
                        display: 'inline-block',
                        width: 10,
                        height: 10,
                        borderRadius: 2,
                        background: categoryColors[cat] || '#8b949e',
                      }} />
                      {categoryLabels[cat] || cat}
                    </div>
                  ))}
                </div>
              )}

              {/* Flame-graph style bar chart */}
              <div style={{
                background: 'var(--bg-tertiary)',
                borderRadius: 'var(--radius)',
                padding: '10px 12px',
                maxHeight: 400,
                overflowY: 'auto',
              }}>
                {nodes.map((node, idx) => {
                  const pct = maxDuration > 0 ? (node.duration_us / maxDuration) * 100 : 0;
                  const widthPct = Math.max(pct, 2); // minimum visible width
                  const color = getOpColor(node.op_name);
                  const isHovered = hoveredNode === idx;

                  return (
                    <div
                      key={node.node_id || idx}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        marginBottom: 3,
                        position: 'relative',
                      }}
                      onMouseEnter={() => setHoveredNode(idx)}
                      onMouseLeave={() => setHoveredNode(null)}
                    >
                      {/* Node label */}
                      <div style={{
                        width: 80,
                        flexShrink: 0,
                        fontSize: 11,
                        fontFamily: 'monospace',
                        color: 'var(--text-secondary)',
                        textAlign: 'right',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}>
                        {node.node_id}
                      </div>

                      {/* Bar */}
                      <div style={{ flex: 1, position: 'relative', height: 22 }}>
                        <div style={{
                          width: `${widthPct}%`,
                          height: '100%',
                          background: isHovered ? color : `${color}cc`,
                          borderRadius: 3,
                          transition: 'width 0.3s ease, background 0.15s ease',
                          display: 'flex',
                          alignItems: 'center',
                          paddingLeft: 6,
                          minWidth: 'fit-content',
                        }}>
                          <span style={{
                            fontSize: 10,
                            fontWeight: 600,
                            color: '#fff',
                            whiteSpace: 'nowrap',
                            textShadow: '0 1px 2px rgba(0,0,0,0.5)',
                          }}>
                            {node.op_name}
                          </span>
                        </div>

                        {/* Tooltip on hover */}
                        {isHovered && (
                          <div style={{
                            position: 'absolute',
                            top: -36,
                            left: Math.min(widthPct, 70) + '%',
                            background: 'var(--bg-primary)',
                            border: '1px solid var(--border)',
                            borderRadius: 'var(--radius)',
                            padding: '4px 8px',
                            fontSize: 11,
                            color: 'var(--text-primary)',
                            whiteSpace: 'nowrap',
                            zIndex: 10,
                            pointerEvents: 'none',
                            boxShadow: 'var(--shadow)',
                          }}>
                            <strong>{node.op_name}</strong> ({node.node_id})
                            &nbsp;&mdash;&nbsp;
                            {formatDuration(node.duration_us)}
                            {totalDuration > 0 && (
                              <span style={{ color: 'var(--text-muted)', marginLeft: 4 }}>
                                ({((node.duration_us / totalDuration) * 100).toFixed(1)}%)
                              </span>
                            )}
                          </div>
                        )}
                      </div>

                      {/* Duration label */}
                      <div style={{
                        width: 70,
                        flexShrink: 0,
                        fontSize: 11,
                        fontFamily: 'monospace',
                        color: 'var(--text-muted)',
                        textAlign: 'right',
                      }}>
                        {formatDuration(node.duration_us)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default NativeProfilePanel;
