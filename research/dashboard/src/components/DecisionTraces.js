import React, { useState, useEffect, useMemo } from 'react';
import { formatTime } from '../utils/format';
import { useAriaData } from '../hooks/useAriaData';
import apiService from '../services/apiService';

/**
 * DecisionTraces — Task 3H
 * 
 * Displays recent decision traces: what was generated/promoted/rejected and why.
 */
export function DecisionTraces() {
  const { slowPollTick } = useAriaData();
  const [decisions, setDecisions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    const fetchDecisions = async () => {
      try {
        const resp = await apiService.getDashboardSummary();

        if (!active) return;

        const source = Array.isArray(resp?.decision_traces)
          ? resp.decision_traces
          : Array.isArray(resp?.recent_decisions)
            ? resp.recent_decisions
            : Array.isArray(resp?.decisions)
              ? resp.decisions
              : [];
        const traces = source.map(d => ({
            id: d.decision_id,
            timestamp: d.timestamp,
            subject: d.subject,
            action: d.decision_type,
            rationale: d.rationale,
            score: d.evidence_pack?.confidence || d.evidence_pack?.total_score,
            top_signal: d.evidence_pack?.top_signal || d.decision_type
          }));

        setDecisions(traces.sort((a, b) => b.timestamp - a.timestamp));
        setError(null);
      } catch (e) {
        if (active) setError('Failed to load decision traces: ' + e.message);
      } finally {
        if (active) setLoading(false);
      }
    };

    fetchDecisions();
    return () => { active = false; };
  }, [slowPollTick]);

  if (loading && decisions.length === 0) {
    return <div className="card ux-state-loading"><span className="ux-spinner" /> Loading traces...</div>;
  }

  if (decisions.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Decision Traces</div>
        <div className="empty-state" style={{ padding: 28 }}>
          <div className="empty-state-title">No decision traces recorded</div>
          <p className="empty-state-hint">
            Automated promotion and rejection traces will appear here once the backend publishes decision events.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">Decision Traces</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 16 }}>
        History of automated research decisions, including promotions, rejections, and plan adjustments.
      </p>

      <div style={{ overflowX: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Candidate / Subject</th>
              <th>Action</th>
              <th>Top Signal</th>
              <th>Score/Conf</th>
            </tr>
          </thead>
          <tbody>
            {decisions.slice(0, 20).map((d) => (
              <tr key={d.id}>
                <td style={{ fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                  {formatTime(d.timestamp)}
                </td>
                <td style={{ fontWeight: 500 }}>
                  {d.subject?.replace('experiment:', '').slice(0, 20)}
                </td>
                <td>
                  <span className={`badge tier-${d.action === 'promote' ? 'validation' : 'screening'}`} style={{ fontSize: 10 }}>
                    {d.action}
                  </span>
                </td>
                <td style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                  {d.top_signal || '—'}
                </td>
                <td style={{ fontWeight: 600 }}>
                  {d.score != null ? (d.score * 100).toFixed(1) + '%' : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default DecisionTraces;
