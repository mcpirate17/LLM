import { apiCall } from "../services/apiService";
import React, { useState, useEffect, useMemo } from 'react';
import FailureAnalysis from './FailureAnalysis';
import ProgramDetail from './ProgramDetail';
import { formatTime, formatDuration } from '../utils/format';
import { lossColor, noveltyColor } from '../utils/colors';
import { candidateScore } from '../utils/scoringEngine';
import { filterRowsByQuery } from '../utils/tableFiltering';


/**
 * ExperimentDetail — Full experiment breakdown with hypothesis, funnel,
 * all programs table, failure analysis, and Aria's LLM analysis.
 */

/** Rate a program row: green (learned well), amber (compiles), red (failed early) */
function programRowRating(p) {
  if (p.stage1_passed) {
    if (p.baseline_loss_ratio != null && p.baseline_loss_ratio < 1.0)
      return { color: 'var(--accent-green)', label: 'Excellent', tip: 'Outperforms a standard transformer of the same size' };
    if ((p.loss_ratio || 1) < 0.5)
      return { color: 'var(--accent-green)', label: 'Strong', tip: 'Learned quickly — loss dropped significantly' };
    return { color: 'var(--accent-yellow)', label: 'Learned', tip: 'Passed Stage 1 — demonstrated learning ability' };
  }
  if (p.stage05_passed)
    return { color: 'var(--accent-orange, #f0883e)', label: 'Stable', tip: 'Numerically stable but didn\'t learn — gradient signal too weak' };
  if (p.stage0_passed)
    return { color: 'var(--accent-orange, #f0883e)', label: 'Compiled', tip: 'Compiled and ran but produced NaN or unstable gradients' };
  return { color: 'var(--accent-red)', label: 'Failed', tip: 'Failed to compile or crashed — invalid operation combination', order: 0 };
}

function progScoreColor(score) {
  if (score >= 70) return 'var(--accent-green)';
  if (score >= 40) return 'var(--accent-yellow)';
  if (score >= 20) return 'var(--accent-orange, #f0883e)';
  return 'var(--accent-red)';
}

const PROG_COLUMNS = [
  { key: '_score', label: 'Utility Score' },
  { key: 'rating', label: 'Rating' },
  { key: 'graph_fingerprint', label: 'Fingerprint' },
  { key: 'stage0_passed', label: 'S0' },
  { key: 'stage05_passed', label: 'S0.5' },
  { key: 'stage1_passed', label: 'S1' },
  { key: 'novelty_score', label: 'Novelty' },
  { key: 'loss_ratio', label: 'Loss Ratio' },
  { key: 'param_count', label: 'Params' },
  { key: 'peak_memory_mb', label: 'Memory' },
  { key: 'flops_forward', label: 'FLOPs' },
  { key: 'baseline_loss_ratio', label: 'Baseline' },
];

const EXPERIMENT_DETAIL_PROGRAM_SORT_PREFS_KEY = 'dashboard.experiment-detail.programs.sort.v1';

const ROW_RATING_ORDER = { Excellent: 5, Strong: 4, Learned: 3, Stable: 2, Compiled: 1, Failed: 0 };

function FunnelViz({ experiment }) {
  const stages = [
    { label: 'Generated', value: experiment.n_programs_generated || 0, color: 'var(--accent-blue)', icon: 'Σ' },
    { label: 'Compiled', value: experiment.n_stage0_passed || 0, color: 'var(--accent-green)', icon: '✓' },
    { label: 'Stable', value: experiment.n_stage05_passed || 0, color: 'var(--accent-yellow)', icon: '±' },
    { label: 'Learned', value: experiment.n_stage1_passed || 0, color: 'var(--accent-purple)', icon: '★' },
  ];

  const max = stages[0].value || 1;

  return (
    <div style={{ display: 'flex', gap: 12, alignItems: 'center', background: 'var(--bg-tertiary)', padding: '16px 20px', borderRadius: 8, border: '1px solid var(--border)' }}>
      {stages.map((stage, i) => {
        const percent = Math.round((stage.value / max) * 100);
        return (
          <React.Fragment key={i}>
            {i > 0 && <div style={{ color: 'var(--text-muted)', fontSize: 20 }}>&rarr;</div>}
            <div style={{ flex: 1, textAlign: 'center' }}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 4 }}>{stage.label}</div>
              <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'center', gap: 4 }}>
                <div style={{ fontSize: 22, fontWeight: 800, color: 'var(--text-primary)' }}>{stage.value}</div>
                {i > 0 && <div style={{ fontSize: 12, color: stage.color }}>{percent}%</div>}
              </div>
              <div style={{ height: 4, background: 'var(--border)', borderRadius: 2, marginTop: 8, overflow: 'hidden' }}>
                <div style={{ width: `${percent}%`, height: '100%', background: stage.color }} />
              </div>
            </div>
          </React.Fragment>
        );
      })}
    </div>
  );
}

function ExperimentSummaryHeader({ experiment, programs }) {
  const bestProgram = useMemo(() => {
    if (!programs || programs.length === 0) return null;
    return [...programs].sort((a, b) => candidateScore(b) - candidateScore(a))[0];
  }, [programs]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16, marginBottom: 16 }}>
      <div className="card" style={{ borderLeft: '4px solid var(--accent-blue)' }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>RESEARCH HYPOTHESIS</div>
        <div style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-primary)', lineHeight: 1.4 }}>
          {experiment.hypothesis || "No hypothesis recorded for this session."}
        </div>
        {experiment.aria_summary && (
          <div style={{ marginTop: 12, fontSize: 13, color: 'var(--text-secondary)', borderTop: '1px solid var(--border)', paddingTop: 12 }}>
            <strong>Outcome:</strong> {experiment.aria_summary}
          </div>
        )}
      </div>

      {bestProgram && (
        <div className="card" style={{ borderLeft: '4px solid var(--accent-green)', background: 'linear-gradient(135deg, var(--bg-secondary) 0%, rgba(63, 185, 80, 0.05) 100%)' }}>
          <div style={{ fontSize: 10, color: 'var(--accent-green)', fontWeight: 700, marginBottom: 4 }}>TOP DISCOVERY</div>
          <div style={{ fontSize: 14, fontWeight: 700, fontFamily: 'monospace' }}>{bestProgram.graph_fingerprint?.slice(0, 12)}</div>
          <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            <div style={{ fontSize: 11 }}>
              <span style={{ color: 'var(--text-muted)' }}>Loss Ratio:</span> 
              <span style={{ color: 'var(--accent-green)', marginLeft: 4, fontWeight: 600 }}>{bestProgram.loss_ratio?.toFixed(4)}</span>
            </div>
            <div style={{ fontSize: 11 }}>
              <span style={{ color: 'var(--text-muted)' }}>Novelty:</span> 
              <span style={{ color: 'var(--accent-purple)', marginLeft: 4, fontWeight: 600 }}>{bestProgram.novelty_score?.toFixed(3)}</span>
            </div>
          </div>
          <div style={{ marginTop: 10, fontSize: 10, color: 'var(--text-muted)' }}>
            Score: <span style={{ color: 'var(--text-primary)', fontWeight: 700 }}>{candidateScore(bestProgram)}</span>
          </div>
        </div>
      )}
    </div>
  );
}

function ProgramsTable({ programs, sortKey, sortDesc, onSort, onSelectProgram }) {
  const [filterQuery, setFilterQuery] = useState('');

  const filtered = useMemo(() => (
    filterRowsByQuery(programs, filterQuery, [
      'graph_fingerprint',
      'result_id',
      'architecture_name',
      'program_id',
      'notes',
    ])
  ), [programs, filterQuery]);

  const sorted = useMemo(() => {
    const aug = filtered.map(p => ({ ...p, _score: candidateScore(p), _rating: programRowRating(p) }));
    aug.sort((a, b) => {
      let va, vb;
      if (sortKey === '_score') { va = a._score; vb = b._score; }
      else if (sortKey === 'rating') { va = ROW_RATING_ORDER[a._rating.label] || 0; vb = ROW_RATING_ORDER[b._rating.label] || 0; }
      else if (sortKey === 'graph_fingerprint') { va = a.graph_fingerprint || ''; vb = b.graph_fingerprint || ''; return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb); }
      else { va = a[sortKey]; vb = b[sortKey]; }
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return sortDesc ? vb - va : va - vb;
    });
    return aug;
  }, [filtered, sortKey, sortDesc]);

  return (
    <div className="card">
      <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
        <span>All Programs ({programs.length})</span>
        <input
          value={filterQuery}
          onChange={(e) => setFilterQuery(e.target.value)}
          placeholder="Filter programs"
          style={{
            fontSize: 11,
            padding: '4px 8px',
            borderRadius: 4,
            border: '1px solid var(--border)',
            background: 'var(--bg-tertiary)',
            color: 'var(--text-primary)',
            minWidth: 160,
          }}
        />
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8, lineHeight: 1.5 }}>
        Every architecture tested in this experiment. P = passed, F = failed at that stage.
        Baseline {'<'} 1.0 means it outperformed a standard transformer of the same size.
        Click any row for the full computation graph and detailed metrics.
      </p>
      <div style={{ maxHeight: 400, overflow: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              {PROG_COLUMNS.map(col => (
                <th
                  key={col.key}
                  onClick={() => onSort(col.key)}
                  style={{ cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap' }}
                >
                  {col.label}
                  {sortKey === col.key && (
                    <span style={{ marginLeft: 4, fontSize: 10 }}>
                      {sortDesc ? '\u25BC' : '\u25B2'}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((p, i) => {
              const rating = p._rating;
              return (
                <tr key={p.result_id || i}
                  style={{ cursor: 'pointer' }}
                  onClick={() => onSelectProgram && onSelectProgram(p.result_id)}>
                  <td style={{ fontWeight: 600, color: progScoreColor(p._score) }}>
                    {p._score}
                  </td>
                  <td title={rating.tip}>
                    <span style={{
                      display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
                      background: rating.color, marginRight: 6,
                    }} />
                    <span style={{ fontSize: 11, color: rating.color }}>{rating.label}</span>
                  </td>
                  <td style={{ fontFamily: 'monospace', fontSize: 12, color: 'var(--accent-blue)' }}>
                    {p.graph_fingerprint?.slice(0, 10) || '--'}
                  </td>
                  <td><span className={`badge ${p.stage0_passed ? 'pass' : 'fail'}`}>{p.stage0_passed ? 'P' : 'F'}</span></td>
                  <td><span className={`badge ${p.stage05_passed ? 'pass' : 'fail'}`}>{p.stage05_passed ? 'P' : 'F'}</span></td>
                  <td><span className={`badge ${p.stage1_passed ? 'pass' : 'fail'}`}>{p.stage1_passed ? 'P' : 'F'}</span></td>
                  <td style={{ color: noveltyColor(p.novelty_score) }}>
                    {p.novelty_score?.toFixed(3) || '--'}
                  </td>
                  <td style={{ color: lossColor(p.loss_ratio) }}>
                    {p.loss_ratio?.toFixed(4) || '--'}
                  </td>
                  <td>{p.param_count ? `${(p.param_count / 1e6).toFixed(1)}M` : '--'}</td>
                  <td style={{ fontSize: 11 }}>{p.peak_memory_mb ? `${Number(p.peak_memory_mb).toFixed(0)}MB` : '--'}</td>
                  <td style={{ fontSize: 11 }}>{p.flops_forward ? `${(p.flops_forward / 1e6).toFixed(1)}M` : '--'}</td>
                  <td style={{
                    fontSize: 11,
                    fontWeight: p.baseline_loss_ratio != null && p.baseline_loss_ratio < 1 ? 600 : 'normal',
                    color: p.baseline_loss_ratio != null
                      ? (p.baseline_loss_ratio < 1 ? 'var(--accent-green)' : 'var(--accent-red)')
                      : 'var(--text-muted)'
                  }}>
                    {p.baseline_loss_ratio?.toFixed(3) || '--'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8, display: 'flex', gap: 16 }}>
        <span><span style={{ color: 'var(--accent-green)' }}>Green</span> = learned from data (S1 pass)</span>
        <span><span style={{ color: 'var(--accent-yellow)' }}>Amber</span> = passed learning stage</span>
        <span><span style={{ color: 'var(--accent-orange, #f0883e)' }}>Orange</span> = compiled but didn't learn</span>
        <span><span style={{ color: 'var(--accent-red)' }}>Red</span> = failed to compile</span>
      </div>
    </div>
  );
}

function ExperimentDetail({ experimentId, onBack, onSelectProgram }) {
  const [data, setData] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedProgramId, setSelectedProgramId] = useState(null);
  const [isRerunning, setIsRerunning] = useState(false);
  const [rerunConfirm, setRerunConfirm] = useState(false);
  const [progSortKey, setProgSortKey] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(EXPERIMENT_DETAIL_PROGRAM_SORT_PREFS_KEY) || '{}');
      const validKeys = new Set(PROG_COLUMNS.map((column) => column.key));
      if (typeof stored.progSortKey === 'string' && validKeys.has(stored.progSortKey)) {
        return stored.progSortKey;
      }
    } catch {}
    return '_score';
  });
  const [progSortDesc, setProgSortDesc] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(EXPERIMENT_DETAIL_PROGRAM_SORT_PREFS_KEY) || '{}');
      if (typeof stored.progSortDesc === 'boolean') {
        return stored.progSortDesc;
      }
    } catch {}
    return true;
  });

  useEffect(() => {
    try {
      localStorage.setItem(
        EXPERIMENT_DETAIL_PROGRAM_SORT_PREFS_KEY,
        JSON.stringify({ progSortKey, progSortDesc }),
      );
    } catch {}
  }, [progSortKey, progSortDesc]);

  useEffect(() => {
    if (!experimentId) return;
    setLoading(true);
    setError(null);
    Promise.all([
      apiCall(`/api/experiments/${experimentId}`).then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      }),
      apiCall(`/api/experiments/${experimentId}/analysis`).then(r => r.json()).catch(() => null),
    ]).then(([expData, analysisData]) => {
      setData(expData);
      setAnalysis(analysisData);
      setLoading(false);
    }).catch(e => {
      const msg = e.message.includes('404')
        ? 'This experiment could not be found. It may have been deleted.'
        : e.message.includes('500')
        ? 'Server error while loading experiment. Try again later.'
        : 'Failed to load experiment: ' + e.message;
      setError(msg);
      setLoading(false);
    });
  }, [experimentId]);

  if (loading) return <div className="card"><p style={{ color: 'var(--text-muted)' }}>Loading experiment...</p></div>;
  if (error) return <div className="card"><p style={{ color: 'var(--accent-red)' }}>{error}</p></div>;
  if (!data || !data.experiment) return <div className="card"><p style={{ color: 'var(--accent-red)' }}>Experiment not found</p></div>;

  const exp = data.experiment;
  const programs = data.programs || [];
  const entries = data.entries || [];
  const prereg = data.preregistration || null;
  const preregDeviations = data.preregistration_deviations || [];

  const handleRerun = async () => {
    if (!rerunConfirm) {
      setRerunConfirm(true);
      setTimeout(() => setRerunConfirm(false), 3000);
      return;
    }
    setIsRerunning(true);
    try {
      const res = await apiCall(`/api/experiments/${experimentId}/rerun`, { method: 'POST' });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error || 'Failed to rerun');
      // On success, go back to command view which shows the active run
      if (onBack) onBack();
    } catch (err) {
      alert(err.message);
    } finally {
      setIsRerunning(false);
      setRerunConfirm(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Header Info */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <button className="refresh-btn" onClick={() => onBack && onBack()} style={{ marginRight: 12 }}>&larr; Back</button>
            <span style={{ fontFamily: 'monospace', color: 'var(--accent-blue)', marginRight: 8 }}>{experimentId}</span>
            <span className={`badge ${exp.status === 'completed' ? 'pass' : exp.status === 'running' ? 'running' : 'fail'}`}>
              {exp.status}
            </span>
            <button
              className="refresh-btn"
              style={{
                marginLeft: 12,
                fontSize: 11,
                borderColor: rerunConfirm ? 'var(--accent-yellow)' : 'var(--border)',
                color: rerunConfirm ? 'var(--accent-yellow)' : 'inherit',
              }}
              disabled={isRerunning || exp.status === 'running'}
              onClick={handleRerun}
            >
              {isRerunning ? 'Starting...' : rerunConfirm ? 'Click to confirm Rerun' : 'Rerun Experiment'}
            </button>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            {formatTime(exp.timestamp)} | {formatDuration(exp.duration_seconds)}
          </div>
        </div>
      </div>

      <ExperimentSummaryHeader experiment={exp} programs={programs} />

      {/* Funnel */}
      <div className="card">
        <div className="card-title">Evaluation Funnel</div>
        <FunnelViz experiment={exp} />
      </div>

      {/* LLM Analysis */}
      {analysis?.analysis && (
        <div className="card">
          <div className="card-title">
            Deep Analysis
            <span className="badge novel" style={{ marginLeft: 8, fontSize: 10 }}>
              {analysis.source === 'stored' ? 'cached' : 'live'}
            </span>
          </div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)', whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
            {analysis.analysis}
          </div>
        </div>
      )}

      {/* Insights */}
      <div className="card">
        <div className="card-title">Insights</div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
          Patterns and takeaways Aria extracted from this experiment's results.
        </p>
        {exp.insights && exp.insights.length > 0 ? (
          exp.insights.map((insight, i) => (
            <div key={i} className="insight-card">
              <div className="insight-content">{insight}</div>
            </div>
          ))
        ) : (
          <p style={{ fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic' }}>
            No insights generated for this experiment yet.
          </p>
        )}
      </div>

      {/* Two-column: Failure Analysis + Programs Table */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 16 }}>
        <FailureAnalysis experimentId={experimentId} />

        {/* Programs Table */}
        <ProgramsTable
          programs={programs}
          sortKey={progSortKey}
          sortDesc={progSortDesc}
          onSort={(key) => {
            if (progSortKey === key) setProgSortDesc(!progSortDesc);
            else { setProgSortKey(key); setProgSortDesc(true); }
          }}
          onSelectProgram={setSelectedProgramId}
        />
      </div>

      {/* Notebook Entries */}
      <div className="card">
        <div className="card-title">Notebook Entries</div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
          Chronological log of observations, decisions, and errors recorded during this experiment.
        </p>
        {entries.length > 0 ? (
          entries.slice(0, 10).map((entry, i) => (
            <div key={i} className={`notebook-entry ${entry.entry_type}`}>
              <div className="entry-header">
                <span className="entry-title">{entry.title}</span>
                <span className="entry-type">{entry.entry_type}</span>
              </div>
              <div className="entry-content">{entry.content}</div>
            </div>
          ))
        ) : (
          <p style={{ fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic' }}>
            No notebook entries recorded for this experiment.
          </p>
        )}
      </div>

      {/* Program Detail Modal */}
      {selectedProgramId && (
        <ProgramDetail
          resultId={selectedProgramId}
          onClose={() => setSelectedProgramId(null)}
        />
      )}
    </div>
  );
}

export default ExperimentDetail;
