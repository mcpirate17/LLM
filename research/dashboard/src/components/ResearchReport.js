import React, { useState, useEffect, useMemo } from 'react';

const API_BASE = process.env.REACT_APP_API_URL || '';

function RatingBadge({ program }) {
  const lr = program.loss_ratio;
  const nov = program.novelty_score || 0;
  const bl = program.baseline_loss_ratio;

  let color, label;
  if (bl != null && bl < 1 && lr < 0.5 && nov > 0.7) {
    color = 'var(--accent-green)'; label = 'S1 - Exceptional';
  } else if (lr < 0.5 && nov > 0.5) {
    color = 'var(--accent-green)'; label = 'S1 - Strong';
  } else if (lr < 0.7) {
    color = 'var(--accent-yellow)'; label = 'S1 - Moderate';
  } else {
    color = 'var(--accent-orange, #f0883e)'; label = 'S1 - Marginal';
  }

  return (
    <span style={{
      padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600,
      background: `${color}22`, color, border: `1px solid ${color}44`,
    }}>
      {label}
    </span>
  );
}

function discoveryScore(p) {
  const lossScore = p.loss_ratio != null ? Math.max(0, 1 - (p.loss_ratio - 0.2) / 0.8) * 35 : 0;
  const noveltyScore = p.novelty_score != null ? Math.min(p.novelty_score, 1.0) * 25 : 0;
  const baselineScore = p.baseline_loss_ratio != null ? Math.max(0, Math.min(1, 1.5 - p.baseline_loss_ratio)) * 30 : 0;
  const similarBonus = p.most_similar_to ? 10 : 0;
  return Math.round(Math.max(0, Math.min(100, lossScore + noveltyScore + baselineScore + similarBonus)));
}

function discScoreColor(score) {
  if (score >= 70) return 'var(--accent-green)';
  if (score >= 40) return 'var(--accent-yellow)';
  if (score >= 20) return 'var(--accent-orange, #f0883e)';
  return 'var(--accent-red)';
}

const DISC_COLUMNS = [
  { key: '_score', label: 'Score' },
  { key: 'graph_fingerprint', label: 'Fingerprint' },
  { key: 'loss_ratio', label: 'Loss Ratio' },
  { key: 'novelty_score', label: 'Novelty' },
  { key: 'baseline_loss_ratio', label: 'Baseline' },
  { key: 'cka_source', label: 'CKA Source' },
  { key: 'most_similar_to', label: 'Similar To' },
  { key: 'rating', label: 'Rating' },
];

const DISC_RATING_ORDER = { 'S1 - Exceptional': 4, 'S1 - Strong': 3, 'S1 - Moderate': 2, 'S1 - Marginal': 1 };

function DiscoveryRankings({ programs }) {
  const [sortKey, setSortKey] = useState('_score');
  const [sortDesc, setSortDesc] = useState(true);

  const handleSort = (key) => {
    if (key === 'rating') key = '_ratingOrder';
    if (sortKey === key) setSortDesc(!sortDesc);
    else { setSortKey(key); setSortDesc(true); }
  };

  const sorted = useMemo(() => {
    const aug = programs.map(p => {
      const lr = p.loss_ratio;
      const nov = p.novelty_score || 0;
      const bl = p.baseline_loss_ratio;
      let rLabel;
      if (bl != null && bl < 1 && lr < 0.5 && nov > 0.7) rLabel = 'S1 - Exceptional';
      else if (lr < 0.5 && nov > 0.5) rLabel = 'S1 - Strong';
      else if (lr < 0.7) rLabel = 'S1 - Moderate';
      else rLabel = 'S1 - Marginal';
      return { ...p, _score: discoveryScore(p), _ratingOrder: DISC_RATING_ORDER[rLabel] || 0 };
    });
    aug.sort((a, b) => {
      let va, vb;
      if (sortKey === 'graph_fingerprint' || sortKey === 'most_similar_to') {
        va = a[sortKey] || ''; vb = b[sortKey] || '';
        return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      }
      va = a[sortKey]; vb = b[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return sortDesc ? vb - va : va - vb;
    });
    return aug;
  }, [programs, sortKey, sortDesc]);

  return (
    <div className="card">
      <div className="card-title">Discovery Rankings</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        The strongest architectures discovered, ranked by a composite of learning speed, novelty, and baseline comparison. Score combines loss ratio (35%), novelty (25%), baseline performance (30%), and identification (10%).
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
              <th style={{ padding: '8px 6px', color: 'var(--text-muted)' }}>#</th>
              {DISC_COLUMNS.map(col => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  style={{ padding: '8px 6px', color: 'var(--text-muted)', cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap' }}
                >
                  {col.label}
                  {(sortKey === col.key || (col.key === 'rating' && sortKey === '_ratingOrder')) && (
                    <span style={{ marginLeft: 4, fontSize: 10 }}>
                      {sortDesc ? '\u25BC' : '\u25B2'}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((p, i) => (
              <tr key={p.result_id || i} style={{ borderBottom: '1px solid var(--border)' }}>
                <td style={{ padding: '6px', color: 'var(--text-muted)' }}>{i + 1}</td>
                <td style={{ padding: '6px', fontWeight: 600, color: discScoreColor(p._score) }}>
                  {p._score}
                </td>
                <td style={{ padding: '6px', fontFamily: 'monospace', color: 'var(--accent-blue)' }}>
                  {(p.graph_fingerprint || '').slice(0, 12)}
                </td>
                <td style={{
                  padding: '6px', fontWeight: 600,
                  color: (p.loss_ratio || 1) < 0.5 ? 'var(--accent-green)' : (p.loss_ratio || 1) < 0.7 ? 'var(--accent-yellow)' : 'var(--text-secondary)',
                }}>
                  {p.loss_ratio != null ? p.loss_ratio.toFixed(4) : '--'}
                </td>
                <td style={{ padding: '6px', color: (p.novelty_score || 0) > 0.7 ? 'var(--accent-green)' : 'var(--text-secondary)' }}>
                  {p.novelty_score != null ? p.novelty_score.toFixed(3) : '--'}
                </td>
                <td style={{
                  padding: '6px',
                  color: p.baseline_loss_ratio != null && p.baseline_loss_ratio < 1 ? 'var(--accent-green)' : 'var(--text-secondary)',
                  fontWeight: p.baseline_loss_ratio != null && p.baseline_loss_ratio < 1 ? 600 : 'normal',
                }}>
                  {p.baseline_loss_ratio != null ? p.baseline_loss_ratio.toFixed(3) : '--'}
                </td>
                <td style={{ padding: '6px' }}>
                  {p.cka_source ? (
                    <span style={{
                      fontSize: 10,
                      fontWeight: 600,
                      padding: '2px 6px',
                      borderRadius: 4,
                      background: p.cka_source === 'artifact' ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)',
                      color: p.cka_source === 'artifact' ? 'var(--accent-green)' : 'var(--accent-red)',
                    }}>
                      {p.cka_source === 'artifact' ? 'artifact' : 'fallback'}
                    </span>
                  ) : '--'}
                  {p.cka_artifact_version && (
                    <span style={{ marginLeft: 6, fontSize: 10, color: 'var(--text-muted)' }}>
                      {p.cka_artifact_version}
                    </span>
                  )}
                </td>
                <td style={{ padding: '6px', color: 'var(--text-muted)', fontSize: 11 }}>
                  {p.most_similar_to || '--'}
                </td>
                <td style={{ padding: '6px' }}>
                  {p.loss_ratio != null && <RatingBadge program={p} />}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function StatCard({ label, value, color }) {
  return (
    <div style={{
      padding: '12px 16px', background: 'var(--bg-tertiary)', borderRadius: 6,
      borderLeft: `3px solid ${color || 'var(--accent-blue)'}`,
    }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: color || 'var(--text-primary)' }}>{value}</div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>{label}</div>
    </div>
  );
}

function EfficiencyChart({ frontier }) {
  if (!frontier || frontier.length === 0) return <p style={{ color: 'var(--text-muted)' }}>No Pareto-optimal programs yet.</p>;

  const W = 500, H = 200;
  const pad = { l: 60, r: 20, t: 20, b: 35 };

  const losses = frontier.map(p => p.final_loss || p.loss_ratio || 0).filter(l => isFinite(l));
  const flops = frontier.map(p => p.flops_forward || p.param_count || 0).filter(f => f > 0);
  if (losses.length < 2 || flops.length < 2) return null;

  const minL = Math.min(...losses), maxL = Math.max(...losses);
  const minF = Math.min(...flops), maxF = Math.max(...flops);
  const rangeL = maxL - minL || 1, rangeF = maxF - minF || 1;

  const xScale = v => pad.l + ((v - minF) / rangeF) * (W - pad.l - pad.r);
  const yScale = v => H - pad.b - ((v - minL) / rangeL) * (H - pad.t - pad.b);

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
      <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke="var(--border)" />
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} stroke="var(--border)" />
      <text x={W / 2} y={H - 5} textAnchor="middle" fill="var(--text-muted)" fontSize={10}>FLOPs / Params</text>
      <text x={12} y={H / 2} textAnchor="middle" fill="var(--text-muted)" fontSize={10} transform={`rotate(-90, 12, ${H / 2})`}>Loss</text>
      {frontier.map((p, i) => {
        const x = xScale(p.flops_forward || p.param_count || 0);
        const y = yScale(p.final_loss || p.loss_ratio || 0);
        if (!isFinite(x) || !isFinite(y)) return null;
        return (
          <circle key={i} cx={x} cy={y} r={5}
            fill="var(--accent-purple)" opacity={0.7}
            stroke="var(--bg-secondary)" strokeWidth={1.5}>
            <title>{p.graph_fingerprint?.slice(0, 10)}: loss={p.final_loss || p.loss_ratio}</title>
          </circle>
        );
      })}
    </svg>
  );
}

function generateMarkdown(data) {
  const s = data.summary || {};
  const lines = [];
  lines.push('# Research Report');
  lines.push(`*Generated: ${new Date().toISOString()}*\n`);

  if (data.narrative) {
    lines.push('## Executive Summary\n');
    lines.push(data.narrative + '\n');
  }

  lines.push('## Key Statistics\n');
  lines.push(`- Total experiments: ${s.total_experiments || 0}`);
  lines.push(`- Programs evaluated: ${s.total_programs_evaluated || 0}`);
  lines.push(`- Stage 1 survivors: ${s.total_s1_passed || 0}`);
  lines.push(`- Novel discoveries: ${s.total_novel || 0}`);
  lines.push('');

  const top = data.top_programs || [];
  if (top.length > 0) {
    lines.push('## Discovery Rankings\n');
    lines.push('| Rank | Fingerprint | Loss Ratio | Novelty | Baseline | Similar To |');
    lines.push('|------|-------------|------------|---------|----------|------------|');
    top.forEach((p, i) => {
      lines.push(
        `| ${i + 1} | \`${(p.graph_fingerprint || '').slice(0, 12)}\` ` +
        `| ${p.loss_ratio != null ? p.loss_ratio.toFixed(4) : '--'} ` +
        `| ${p.novelty_score != null ? p.novelty_score.toFixed(3) : '--'} ` +
        `| ${p.baseline_loss_ratio != null ? p.baseline_loss_ratio.toFixed(3) : '--'} ` +
        `| ${p.most_similar_to || '--'} |`
      );
    });
    lines.push('');
  }

  const ops = data.op_success_rates || [];
  if (ops.length > 0) {
    lines.push('## Op Success Rates\n');
    lines.push('| Op | S1 Rate | Count |');
    lines.push('|----|---------|-------|');
    (Array.isArray(ops) ? ops : []).slice(0, 20).forEach(op => {
      lines.push(`| ${op.op_name || '?'} | ${op.s1_rate != null ? (op.s1_rate * 100).toFixed(1) + '%' : '--'} | ${op.total_count || '--'} |`);
    });
    lines.push('');
  }

  const failures = data.failure_patterns || {};
  if (Object.keys(failures).length > 0) {
    lines.push('## Failure Patterns\n');
    lines.push('```json');
    lines.push(JSON.stringify(failures, null, 2));
    lines.push('```\n');
  }

  const insights = data.insights || [];
  if (insights.length > 0) {
    lines.push('## Insights\n');
    insights.forEach(ins => {
      lines.push(`- **[${ins.category || 'general'}]** ${ins.content || ins}`);
    });
    lines.push('');
  }

  return lines.join('\n');
}

function ResearchReport() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/api/report`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(d => { setData(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const handleExport = () => {
    if (!data) return;
    const md = generateMarkdown(data);
    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `research_report_${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (loading) return <div className="card"><p style={{ color: 'var(--text-muted)' }}>Loading report...</p></div>;
  if (error) return <div className="card"><p style={{ color: 'var(--accent-red)' }}>Error loading report: {error}</p></div>;
  if (!data) return null;

  const s = data.summary || {};
  const top = data.top_programs || [];
  const experiments = data.recent_experiments || [];
  const ops = data.op_success_rates || [];
  const failures = data.failure_patterns || {};
  const frontier = data.efficiency_frontier || [];
  const grammarWeights = data.grammar_weights || {};
  const insights = data.insights || [];
  const learningLog = data.learning_log || [];

  const totalProg = s.total_programs_evaluated || 0;
  const s1Rate = totalProg > 0 ? ((s.total_s1_passed || 0) / totalProg * 100).toFixed(1) : '0.0';

  // Separate best and worst ops
  const sortedOps = Array.isArray(ops)
    ? [...ops].sort((a, b) => (b.s1_rate || 0) - (a.s1_rate || 0))
    : [];
  const bestOps = sortedOps.filter(op => (op.s1_rate || 0) > 0).slice(0, 10);
  const worstOps = sortedOps.filter(op => (op.s1_rate || 0) === 0 && (op.total_count || 0) > 5).slice(0, 10);

  // Failure breakdown
  const failureByType = failures.by_error_type || failures;
  const failureByStage = failures.by_stage || {};

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Header + Export */}
      <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div className="card-title" style={{ marginBottom: 4 }}>Research Report</div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            Consolidated findings from {s.total_experiments || 0} experiments
          </div>
        </div>
        <button className="start-btn" onClick={handleExport} style={{ padding: '8px 16px', fontSize: 13 }}>
          Export Markdown
        </button>
      </div>

      {/* Executive Summary */}
      <div className="card">
        <div className="card-title">Executive Summary</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 12, marginBottom: 16 }}>
          <StatCard label="Experiments" value={s.total_experiments || 0} color="var(--accent-blue)" />
          <StatCard label="Programs Tested" value={totalProg.toLocaleString()} color="var(--accent-purple)" />
          <StatCard label="S1 Survivors" value={s.total_s1_passed || 0} color="var(--accent-green)" />
          <StatCard label="S1 Pass Rate" value={`${s1Rate}%`} color={parseFloat(s1Rate) > 5 ? 'var(--accent-green)' : 'var(--accent-yellow)'} />
          <StatCard label="Novel" value={s.total_novel || 0} color="var(--accent-yellow)" />
        </div>
        {data.narrative && (
          <div style={{
            padding: 16, background: 'var(--bg-tertiary)', borderRadius: 6,
            borderLeft: '3px solid var(--accent-purple)', fontSize: 13,
            lineHeight: 1.6, color: 'var(--text-secondary)', whiteSpace: 'pre-wrap',
          }}>
            <div style={{ fontSize: 11, color: 'var(--accent-purple)', fontWeight: 600, marginBottom: 8, textTransform: 'uppercase' }}>
              Aria's Narrative
            </div>
            {data.narrative}
          </div>
        )}
      </div>

      {/* Discovery Rankings */}
      {top.length > 0 && (
        <DiscoveryRankings programs={top} />
      )}

      {/* Experiment Timeline */}
      {experiments.length > 0 && (
        <div className="card">
          <div className="card-title">Experiment Timeline</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
            Chronological view of experiments showing how pass rates and discovery quality evolved over the search.
          </p>
          <div style={{ maxHeight: 400, overflowY: 'auto' }}>
            {experiments.map((exp, i) => {
              const s1 = exp.n_stage1_passed || 0;
              const total = exp.n_programs || 0;
              const confirmed = s1 > 0;
              return (
                <div key={exp.experiment_id || i} style={{
                  padding: '8px 12px', borderBottom: '1px solid var(--border)',
                  display: 'flex', gap: 12, alignItems: 'center',
                }}>
                  <span style={{
                    width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                    background: confirmed ? 'var(--accent-green)' : 'var(--accent-red)',
                  }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 12, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {exp.hypothesis ? `"${exp.hypothesis.slice(0, 80)}"` : `Experiment ${exp.experiment_id?.slice(0, 8)}`}
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      {exp.experiment_type || 'synthesis'} | {total} programs | {s1} S1 | {exp.created_at?.slice(0, 16)}
                    </div>
                  </div>
                  <span style={{
                    fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 4,
                    background: confirmed ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)',
                    color: confirmed ? 'var(--accent-green)' : 'var(--accent-red)',
                  }}>
                    {confirmed ? 'Confirmed' : 'Refuted'}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* What Works + What Doesn't Work */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div className="card">
          <div className="card-title">What Works</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
            Operation types and patterns that consistently appear in successful architectures that passed Stage 1 learning evaluation.
          </p>
          {bestOps.length > 0 ? (
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Top Performing Ops</div>
              {bestOps.map((op, i) => (
                <div key={op.op_name || i} style={{
                  display: 'flex', justifyContent: 'space-between', padding: '4px 0',
                  borderBottom: '1px solid var(--border)',
                }}>
                  <span style={{ fontSize: 12, fontFamily: 'monospace' }}>{op.op_name}</span>
                  <span style={{ fontSize: 12, color: 'var(--accent-green)', fontWeight: 600 }}>
                    {((op.s1_rate || 0) * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          ) : <p style={{ color: 'var(--text-muted)', fontSize: 12 }}>Insufficient data</p>}

          {data.structural_correlations && Object.keys(data.structural_correlations).length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Structural Correlations</div>
              {Object.entries(data.structural_correlations)
                .filter(([, v]) => Math.abs(v) > 0.1)
                .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                .slice(0, 8)
                .map(([key, val]) => (
                  <div key={key} style={{
                    display: 'flex', justifyContent: 'space-between', padding: '3px 0',
                    borderBottom: '1px solid var(--border)',
                  }}>
                    <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{key}</span>
                    <span style={{
                      fontSize: 11, fontWeight: 600,
                      color: val > 0 ? 'var(--accent-green)' : 'var(--accent-red)',
                    }}>
                      {val > 0 ? '+' : ''}{val.toFixed(3)}
                    </span>
                  </div>
                ))}
            </div>
          )}

          {data.top_op_combinations && data.top_op_combinations.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Best Op Combinations</div>
              {data.top_op_combinations.slice(0, 5).map((combo, i) => (
                <div key={i} style={{ fontSize: 11, padding: '3px 0', borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
                  {combo.ops ? combo.ops.join(' + ') : JSON.stringify(combo)}
                  {combo.s1_rate != null && (
                    <span style={{ marginLeft: 8, color: 'var(--accent-green)', fontWeight: 600 }}>
                      {(combo.s1_rate * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="card">
          <div className="card-title">What Doesn't Work</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
            Operation types and patterns that consistently lead to failure — compilation errors, numerical instability, or inability to learn.
          </p>
          {Object.keys(failureByType).length > 0 || Object.keys(failureByStage).length > 0 ? (
            <>
              {Object.keys(failureByStage).length > 0 && (
                <div style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Failures by Stage</div>
                  {Object.entries(failureByStage).map(([stage, count]) => (
                    <div key={stage} style={{
                      display: 'flex', justifyContent: 'space-between', padding: '3px 0',
                      borderBottom: '1px solid var(--border)',
                    }}>
                      <span style={{ fontSize: 12 }}>{stage}</span>
                      <span style={{ fontSize: 12, color: 'var(--accent-red)' }}>{count}</span>
                    </div>
                  ))}
                </div>
              )}
              {typeof failureByType === 'object' && !Array.isArray(failureByType) && Object.keys(failureByType).length > 0 && (
                <div style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Failures by Error Type</div>
                  {Object.entries(failureByType).slice(0, 10).map(([errType, count]) => (
                    <div key={errType} style={{
                      display: 'flex', justifyContent: 'space-between', padding: '3px 0',
                      borderBottom: '1px solid var(--border)', gap: 8,
                    }}>
                      <span style={{ fontSize: 11, color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{errType}</span>
                      <span style={{ fontSize: 11, color: 'var(--accent-red)', flexShrink: 0 }}>{typeof count === 'number' ? count : JSON.stringify(count)}</span>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : <p style={{ color: 'var(--text-muted)', fontSize: 12 }}>No failure data yet</p>}

          {worstOps.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Worst Performing Ops (0% S1)</div>
              {worstOps.map((op, i) => (
                <div key={op.op_name || i} style={{
                  display: 'flex', justifyContent: 'space-between', padding: '3px 0',
                  borderBottom: '1px solid var(--border)',
                }}>
                  <span style={{ fontSize: 12, fontFamily: 'monospace', color: 'var(--text-secondary)' }}>{op.op_name}</span>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{op.total_count} uses</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Grammar Evolution */}
      {grammarWeights.learned && grammarWeights.default && (
        <div className="card">
          <div className="card-title">Grammar Evolution</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
            How the generation weights shifted over time. Rising bars mean the system generates more of that operation; falling bars mean it learned to avoid it.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Weight Changes</div>
              {Object.keys({ ...grammarWeights.default, ...grammarWeights.learned }).sort().map(cat => {
                const old_w = grammarWeights.default[cat] || 1.0;
                const new_w = grammarWeights.learned ? (grammarWeights.learned[cat] || old_w) : old_w;
                const changed = Math.abs(new_w - old_w) > 0.1;
                return (
                  <div key={cat} style={{
                    display: 'flex', justifyContent: 'space-between', padding: '3px 0',
                    borderBottom: '1px solid var(--border)',
                    opacity: changed ? 1 : 0.5,
                  }}>
                    <span style={{ fontSize: 12 }}>{cat}</span>
                    <span style={{ fontSize: 12 }}>
                      <span style={{ color: 'var(--text-muted)' }}>{old_w.toFixed(1)}</span>
                      {changed && (
                        <>
                          <span style={{ color: 'var(--text-muted)', margin: '0 4px' }}>&rarr;</span>
                          <span style={{
                            fontWeight: 600,
                            color: new_w > old_w ? 'var(--accent-green)' : 'var(--accent-red)',
                          }}>
                            {new_w.toFixed(1)}
                          </span>
                        </>
                      )}
                    </span>
                  </div>
                );
              })}
            </div>
            {learningLog.length > 0 && (
              <div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Recent Weight Changes</div>
                <div style={{ maxHeight: 200, overflowY: 'auto' }}>
                  {learningLog.slice(0, 10).map((entry, i) => (
                    <div key={i} style={{ padding: '4px 0', borderBottom: '1px solid var(--border)', fontSize: 11 }}>
                      <div style={{ color: 'var(--text-secondary)' }}>{entry.description || entry.event_type}</div>
                      <div style={{ color: 'var(--text-muted)' }}>{entry.created_at?.slice(0, 16)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Efficiency Frontier */}
      {frontier.length > 0 && (
        <div className="card">
          <div className="card-title">Efficiency Frontier</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
            Trade-off between model size (parameters) and learning speed (loss ratio). Points on the frontier are the best architectures at each size — nothing else learns faster for the same parameter budget.
          </p>
          <EfficiencyChart frontier={frontier} />
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
            {frontier.length} Pareto-optimal programs (lower loss, fewer FLOPs = better)
          </div>
        </div>
      )}

      {/* Insights / Recommendations */}
      {insights.length > 0 && (
        <div className="card">
          <div className="card-title">Insights & Recommendations</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
            Key takeaways and suggested next steps synthesized from all experiments.
          </p>
          {insights.slice(0, 15).map((ins, i) => (
            <div key={i} style={{
              padding: '8px 12px', borderBottom: '1px solid var(--border)',
              display: 'flex', gap: 8, alignItems: 'flex-start',
            }}>
              <span style={{
                fontSize: 10, fontWeight: 600, padding: '2px 6px', borderRadius: 3,
                background: 'var(--bg-tertiary)', color: 'var(--text-muted)',
                textTransform: 'uppercase', flexShrink: 0,
              }}>
                {ins.category || 'insight'}
              </span>
              <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                {ins.content || (typeof ins === 'string' ? ins : JSON.stringify(ins))}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default ResearchReport;
