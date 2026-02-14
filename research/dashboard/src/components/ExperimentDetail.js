import React, { useState, useEffect } from 'react';
import FailureAnalysis from './FailureAnalysis';
import ProgramDetail from './ProgramDetail';

const API_BASE = process.env.REACT_APP_API_URL || '';

/**
 * ExperimentDetail — Full experiment breakdown with hypothesis, funnel,
 * all programs table, failure analysis, and Aria's LLM analysis.
 */

function formatTime(timestamp) {
  if (!timestamp) return '--';
  return new Date(timestamp * 1000).toLocaleString();
}

function formatDuration(seconds) {
  if (!seconds) return '--';
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

function FunnelViz({ experiment }) {
  const stages = [
    { label: 'Generated', value: experiment.n_programs_generated || 0, color: 'var(--accent-blue)' },
    { label: 'S0 Pass', value: experiment.n_stage0_passed || 0, color: 'var(--accent-green)' },
    { label: 'S0.5 Pass', value: experiment.n_stage05_passed || 0, color: 'var(--accent-yellow)' },
    { label: 'S1 Pass', value: experiment.n_stage1_passed || 0, color: 'var(--accent-purple)' },
  ];

  const max = stages[0].value || 1;

  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'end' }}>
      {stages.map((stage, i) => {
        const height = Math.max((stage.value / max) * 60, 4);
        return (
          <div key={i} style={{ flex: 1, textAlign: 'center' }}>
            <div style={{ fontSize: 16, fontWeight: 700, color: stage.color }}>{stage.value}</div>
            <div style={{
              height,
              background: stage.color,
              opacity: 0.3,
              borderRadius: '4px 4px 0 0',
              margin: '4px auto',
              width: '80%',
            }} />
            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{stage.label}</div>
          </div>
        );
      })}
    </div>
  );
}

function ExperimentDetail({ experimentId, onBack, onSelectProgram }) {
  const [data, setData] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedProgramId, setSelectedProgramId] = useState(null);

  useEffect(() => {
    if (!experimentId) return;
    setLoading(true);
    Promise.all([
      fetch(`${API_BASE}/api/experiments/${experimentId}`).then(r => r.json()),
      fetch(`${API_BASE}/api/experiments/${experimentId}/analysis`).then(r => r.json()).catch(() => null),
    ]).then(([expData, analysisData]) => {
      setData(expData);
      setAnalysis(analysisData);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, [experimentId]);

  if (loading) return <div className="card"><p style={{ color: 'var(--text-muted)' }}>Loading experiment...</p></div>;
  if (!data || !data.experiment) return <div className="card"><p style={{ color: 'var(--accent-red)' }}>Experiment not found</p></div>;

  const exp = data.experiment;
  const programs = data.programs || [];
  const entries = data.entries || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Header */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <div>
            <button className="refresh-btn" onClick={onBack} style={{ marginRight: 12 }}>&larr; Back</button>
            <span style={{ fontFamily: 'monospace', color: 'var(--accent-blue)' }}>{experimentId}</span>
            <span className={`badge ${exp.status === 'completed' ? 'pass' : exp.status === 'running' ? 'running' : 'fail'}`}
              style={{ marginLeft: 8 }}>
              {exp.status}
            </span>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            {formatTime(exp.timestamp)} | {formatDuration(exp.duration_seconds)}
          </div>
        </div>

        {/* Hypothesis */}
        {exp.hypothesis && (
          <div style={{
            fontStyle: 'italic',
            color: 'var(--text-secondary)',
            fontSize: 13,
            padding: 8,
            background: 'var(--bg-tertiary)',
            borderRadius: 4,
            marginBottom: 12,
          }}>
            {exp.hypothesis}
          </div>
        )}

        {/* Funnel */}
        <FunnelViz experiment={exp} />
      </div>

      {/* Aria Summary */}
      {exp.aria_summary && (
        <div className="card">
          <div className="card-title">Aria's Summary</div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)', whiteSpace: 'pre-wrap' }}>
            {exp.aria_summary}
          </div>
        </div>
      )}

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
      {exp.insights && exp.insights.length > 0 && (
        <div className="card">
          <div className="card-title">Insights</div>
          {exp.insights.map((insight, i) => (
            <div key={i} className="insight-card">
              <div className="insight-content">{insight}</div>
            </div>
          ))}
        </div>
      )}

      {/* Two-column: Failure Analysis + Programs Table */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 16 }}>
        <FailureAnalysis experimentId={experimentId} />

        {/* Programs Table */}
        <div className="card">
          <div className="card-title">All Programs ({programs.length})</div>
          <div style={{ maxHeight: 400, overflow: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Fingerprint</th>
                  <th>S0</th>
                  <th>S0.5</th>
                  <th>S1</th>
                  <th>Novelty</th>
                  <th>Loss Ratio</th>
                  <th>Params</th>
                </tr>
              </thead>
              <tbody>
                {programs.map((p, i) => (
                  <tr key={p.result_id || i}
                    style={{ cursor: 'pointer' }}
                    onClick={() => setSelectedProgramId(p.result_id)}>
                    <td style={{ fontFamily: 'monospace', fontSize: 12, color: 'var(--accent-blue)' }}>
                      {p.graph_fingerprint?.slice(0, 10) || '--'}
                    </td>
                    <td><span className={`badge ${p.stage0_passed ? 'pass' : 'fail'}`}>{p.stage0_passed ? 'P' : 'F'}</span></td>
                    <td><span className={`badge ${p.stage05_passed ? 'pass' : 'fail'}`}>{p.stage05_passed ? 'P' : 'F'}</span></td>
                    <td><span className={`badge ${p.stage1_passed ? 'pass' : 'fail'}`}>{p.stage1_passed ? 'P' : 'F'}</span></td>
                    <td>
                      <span className={p.novelty_score > 0.5 ? 'badge novel' : ''}>
                        {p.novelty_score?.toFixed(3) || '--'}
                      </span>
                    </td>
                    <td>{p.loss_ratio?.toFixed(4) || '--'}</td>
                    <td>{p.param_count ? `${(p.param_count / 1e6).toFixed(1)}M` : '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Notebook Entries */}
      {entries.length > 0 && (
        <div className="card">
          <div className="card-title">Notebook Entries</div>
          {entries.slice(0, 10).map((entry, i) => (
            <div key={i} className={`notebook-entry ${entry.entry_type}`}>
              <div className="entry-header">
                <span className="entry-title">{entry.title}</span>
                <span className="entry-type">{entry.entry_type}</span>
              </div>
              <div className="entry-content">{entry.content}</div>
            </div>
          ))}
        </div>
      )}

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
