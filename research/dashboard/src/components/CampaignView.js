import React, { useState, useEffect } from 'react';
import useCopyToClipboard from '../hooks/useCopyToClipboard';

const API_BASE = process.env.REACT_APP_API_URL || '';

const STATUS_COLORS = {
  active: 'var(--accent-green)',
  paused: 'var(--accent-yellow)',
  completed: 'var(--accent-blue)',
  abandoned: 'var(--accent-red)',
};

const HYPOTHESIS_COLORS = {
  confirmed: 'var(--accent-green)',
  refuted: 'var(--accent-red)',
  inconclusive: 'var(--accent-yellow)',
  pending: 'var(--text-muted)',
  testing: 'var(--accent-blue)',
};

const DECISION_COLORS = {
  go: 'var(--accent-green)',
  no_go: 'var(--accent-red)',
  pivot: 'var(--accent-yellow)',
  escalate: 'var(--accent-blue)',
  abandon: 'var(--accent-red)',
};

function StatusBadge({ status, colors }) {
  return (
    <span style={{
      padding: '2px 8px',
      borderRadius: 4,
      fontSize: 11,
      fontWeight: 600,
      color: colors[status] || 'var(--text-muted)',
      background: `${colors[status] || 'var(--text-muted)'}22`,
      border: `1px solid ${colors[status] || 'var(--border)'}`,
      textTransform: 'uppercase',
    }}>
      {status?.replace('_', ' ')}
    </span>
  );
}

function CampaignList({ onSelectCampaign }) {
  const [campaigns, setCampaigns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/campaigns`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(d => {
        setCampaigns(Array.isArray(d) ? d : []);
        setLastUpdated(new Date());
        setLoading(false);
      })
      .catch(e => { setError('Failed to load campaigns: ' + e.message); setLoading(false); });
  }, []);

  if (loading) return <p style={{ color: 'var(--text-muted)' }}>Loading campaigns...</p>;
  if (error) return <p style={{ color: 'var(--accent-red)' }}>{error}</p>;
  if (campaigns.length === 0) return <p style={{ color: 'var(--text-muted)' }}>No campaigns yet. Start a continuous experiment to auto-create one.</p>;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
        Last updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : 'loading'} · Source: /api/campaigns
      </div>
      {campaigns.map(c => (
        <div
          key={c.campaign_id}
          className="card"
          style={{ cursor: 'pointer', padding: 16 }}
          role="button"
          tabIndex={0}
          aria-label={`Open campaign ${c.title || c.campaign_id}`}
          onClick={() => onSelectCampaign(c.campaign_id)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              onSelectCampaign(c.campaign_id);
            }
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <h3 style={{ margin: 0, fontSize: 15 }}>{c.title}</h3>
            <StatusBadge status={c.status} colors={STATUS_COLORS} />
          </div>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)', margin: '4px 0' }}>{c.objective}</p>
          <div style={{ display: 'flex', gap: 16, fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>
            <span>{c.n_experiments || 0} experiments</span>
            <span>{c.n_hypotheses || 0} hypotheses</span>
            <span>{c.n_decisions || 0} decisions</span>
          </div>
          <div style={{ marginTop: 10, display: 'flex', justifyContent: 'flex-end' }}>
            <button
              className="refresh-btn"
              style={{ fontSize: 11, padding: '4px 10px' }}
              onClick={(e) => {
                e.stopPropagation();
                onSelectCampaign(c.campaign_id);
              }}
              aria-label={`Open details for campaign ${c.title || c.campaign_id}`}
            >
              Open Details
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

function HypothesisChain({ hypotheses }) {
  if (!hypotheses || hypotheses.length === 0) return null;

  return (
    <div style={{ position: 'relative', paddingLeft: 20 }}>
      {/* Vertical line */}
      <div style={{
        position: 'absolute', left: 8, top: 0, bottom: 0,
        width: 2, background: 'var(--border)',
      }} />

      {hypotheses.map((h, i) => (
        <div key={h.hypothesis_id} style={{ position: 'relative', marginBottom: 16 }}>
          {/* Node dot */}
          <div style={{
            position: 'absolute', left: -16, top: 4,
            width: 12, height: 12, borderRadius: '50%',
            background: HYPOTHESIS_COLORS[h.status] || 'var(--text-muted)',
            border: '2px solid var(--bg-primary)',
          }} />

          <div style={{
            padding: 12, background: 'var(--bg-secondary)', borderRadius: 6,
            border: `1px solid ${HYPOTHESIS_COLORS[h.status] || 'var(--border)'}44`,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
              <StatusBadge status={h.status} colors={HYPOTHESIS_COLORS} />
              {h.confidence_before != null && (
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  Confidence: {(h.confidence_before * 100).toFixed(0)}%
                  {h.confidence_after != null && ` → ${(h.confidence_after * 100).toFixed(0)}%`}
                </span>
              )}
            </div>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
              {h.prediction}
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              <em>Because:</em> {h.reasoning}
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
              <em>Test:</em> {h.test_method} | <em>Metric:</em> {h.success_metric}
            </div>
            {h.outcome_summary && (
              <div style={{
                fontSize: 12, marginTop: 8, padding: 8,
                background: 'var(--bg-tertiary)', borderRadius: 4,
                borderLeft: `2px solid ${HYPOTHESIS_COLORS[h.status]}`,
              }}>
                {h.outcome_summary}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

function parseJsonArray(value) {
  if (Array.isArray(value)) return value;
  if (typeof value !== 'string' || value.trim() === '') return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function DecisionLog({ decisions, hypotheses, experiments, onSelectExperiment }) {
  if (!decisions || decisions.length === 0) return null;

  const hypothesisById = new Map((hypotheses || []).map(h => [h.hypothesis_id, h]));
  const experimentById = new Map((experiments || []).map(exp => [exp.experiment_id, exp]));
  const [copiedValue, copyText] = useCopyToClipboard();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {decisions.map(d => {
        const evidenceIds = parseJsonArray(d.evidence_ids);
        const alternatives = parseJsonArray(d.alternatives_considered);
        const linkedHypotheses = evidenceIds
          .map(id => hypothesisById.get(id))
          .filter(Boolean);
        const linkedExperiments = evidenceIds
          .map(id => experimentById.get(id))
          .filter(Boolean);

        return (
          <div key={d.decision_id} style={{
            padding: 10, background: 'var(--bg-secondary)', borderRadius: 4,
            borderLeft: `3px solid ${DECISION_COLORS[d.decision_type] || 'var(--border)'}`,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <StatusBadge status={d.decision_type} colors={DECISION_COLORS} />
              <span style={{ fontSize: 13, fontWeight: 500 }}>{d.subject}</span>
              {d.decision_id && (
                <>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                    {d.decision_id}
                  </span>
                  <button
                    className="refresh-btn"
                    style={{ fontSize: 10, padding: '1px 6px' }}
                    onClick={() => copyText(d.decision_id)}
                    aria-label={`Copy decision id ${d.decision_id}`}
                  >
                    {copiedValue === d.decision_id ? 'Copied ID' : 'Copy ID'}
                  </button>
                </>
              )}
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{d.rationale}</div>

            <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
              <div>
                <strong>Evidence IDs:</strong> {evidenceIds.length > 0 ? evidenceIds.length : 'not linked'}
                {evidenceIds.length > 0 && (
                  <button
                    className="refresh-btn"
                    style={{ fontSize: 10, padding: '1px 6px', marginLeft: 8 }}
                    onClick={() => copyText(evidenceIds.join(','))}
                    aria-label={`Copy evidence ids for decision ${d.decision_id}`}
                  >
                    {copiedValue === evidenceIds.join(',') ? 'Copied Evidence' : 'Copy Evidence IDs'}
                  </button>
                )}
              </div>
              <div>
                <strong>Linked hypotheses:</strong> {linkedHypotheses.length > 0 ? linkedHypotheses.length : 'none matched'}
              </div>
              <div>
                <strong>Linked experiments:</strong> {linkedExperiments.length > 0 ? linkedExperiments.length : 'none matched'}
              </div>
            </div>

            {linkedHypotheses.length > 0 && (
              <div style={{ marginTop: 8, padding: 8, borderRadius: 4, background: 'var(--bg-tertiary)' }}>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
                  Supporting hypotheses
                </div>
                {linkedHypotheses.map(h => (
                  <div key={h.hypothesis_id} style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>
                    <span style={{ fontFamily: 'monospace', marginRight: 6 }}>{h.hypothesis_id}</span>
                    {h.prediction || 'No prediction text'}
                    {h.status && (
                      <span style={{ marginLeft: 6 }}>
                        <StatusBadge status={h.status} colors={HYPOTHESIS_COLORS} />
                      </span>
                    )}
                  </div>
                ))}
              </div>
            )}

            {linkedExperiments.length > 0 && (
              <div style={{ marginTop: 8, padding: 8, borderRadius: 4, background: 'var(--bg-tertiary)' }}>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
                  Supporting experiments & observed metrics
                </div>
                {linkedExperiments.map(exp => (
                  <div key={exp.experiment_id} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4, flexWrap: 'wrap' }}>
                    <span style={{ fontSize: 11, fontFamily: 'monospace', color: 'var(--text-secondary)' }}>
                      {exp.experiment_id}
                    </span>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      S1 {exp.n_stage1_passed || 0}/{exp.n_programs_generated || 0}
                    </span>
                    {exp.best_loss_ratio != null && (
                      <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                        best loss {exp.best_loss_ratio.toFixed(4)}
                      </span>
                    )}
                    {exp.best_novelty_score != null && (
                      <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                        novelty {exp.best_novelty_score.toFixed(3)}
                      </span>
                    )}
                    {onSelectExperiment && exp.experiment_id && (
                      <button
                        className="refresh-btn"
                        style={{ fontSize: 10, padding: '2px 7px' }}
                        onClick={() => onSelectExperiment(exp.experiment_id)}
                        aria-label={`Open supporting experiment ${exp.experiment_id}`}
                      >
                        Open Experiment
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}

            {alternatives.length > 0 && (
              <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)' }}>
                <strong>Alternatives considered:</strong> {alternatives.length}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function CampaignDetail({ campaignId, onBack, onSelectExperiment }) {
  const [data, setData] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [activeSection, setActiveSection] = useState('timeline');
  const [error, setError] = useState(null);
  const [reportError, setReportError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [reportGeneratedAt, setReportGeneratedAt] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/campaigns/${campaignId}`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(d => {
        setData(d);
        setLastUpdated(new Date());
        setLoading(false);
      })
      .catch(e => { setError('Failed to load campaign: ' + e.message); setLoading(false); });
  }, [campaignId]);

  const generateReport = async () => {
    setGenerating(true);
    setReportError(null);
    try {
      const r = await fetch(`${API_BASE}/api/campaigns/${campaignId}/report`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      setReport(d);
      setReportGeneratedAt(new Date());
    } catch (e) {
      setReportError('Failed to generate report: ' + e.message);
    }
    setGenerating(false);
  };

  if (loading) return <p style={{ color: 'var(--text-muted)' }}>Loading...</p>;
  if (error) return <p style={{ color: 'var(--accent-red)' }}>{error}</p>;
  if (!data) return <p style={{ color: 'var(--accent-red)' }}>Campaign not found</p>;

  const campaign = data.campaign || {};
  const experiments = Array.isArray(data.experiments) ? data.experiments : [];
  const hypotheses = Array.isArray(data.hypotheses) ? data.hypotheses : [];
  const decisions = Array.isArray(data.decisions) ? data.decisions : [];
  const confirmed = hypotheses.filter(h => h.status === 'confirmed').length;
  const refuted = hypotheses.filter(h => h.status === 'refuted').length;
  const resolvedHypotheses = confirmed + refuted;
  const pendingHypotheses = hypotheses.filter(h => h.status !== 'confirmed' && h.status !== 'refuted').length;
  const experimentsWithEvidence = experiments.filter(exp => ((exp.n_programs_generated || exp.n_programs || 0) > 0)).length;
  const evidenceCoveragePct = experiments.length > 0 ? Math.round((experimentsWithEvidence / experiments.length) * 100) : 0;
  const hypothesisResolutionPct = hypotheses.length > 0 ? Math.round((resolvedHypotheses / hypotheses.length) * 100) : 0;
  const decisionCoveragePct = resolvedHypotheses > 0
    ? Math.min(100, Math.round((decisions.length / resolvedHypotheses) * 100))
    : (decisions.length > 0 ? 100 : 0);
  const progressSignals = [
    {
      label: 'Evidence Coverage',
      pct: evidenceCoveragePct,
      detail: `${experimentsWithEvidence}/${experiments.length || 0} experiments produced measurable evidence`,
    },
    {
      label: 'Hypothesis Resolution',
      pct: hypothesisResolutionPct,
      detail: `${resolvedHypotheses}/${hypotheses.length || 0} hypotheses resolved (confirmed/refuted)`,
    },
    {
      label: 'Decision Coverage',
      pct: decisionCoveragePct,
      detail: `${decisions.length}/${resolvedHypotheses || 0} decisions linked to resolved hypotheses`,
    },
  ];
  const blockers = [
    experiments.length === 0 ? 'No experiments run yet for this objective.' : null,
    hypotheses.length === 0 ? 'No explicit hypotheses are captured from current evidence.' : null,
    pendingHypotheses > 0 ? `${pendingHypotheses} hypotheses are still pending outcome.` : null,
    decisions.length === 0 && hypotheses.length > 0 ? 'No go/no-go decision recorded yet.' : null,
  ].filter(Boolean);
  const campaignHealth = experiments.length === 0
    ? { label: 'Not Started', color: 'var(--accent-red)' }
    : blockers.length >= 2
      ? { label: 'At Risk', color: 'var(--accent-yellow)' }
      : pendingHypotheses === 0 && decisions.length > 0
        ? { label: 'Decision-Ready', color: 'var(--accent-green)' }
        : { label: 'In Progress', color: 'var(--accent-blue)' };
  const statusMeaning = campaign.status === 'active'
    ? 'Actively collecting evidence from new experiments.'
    : campaign.status === 'paused'
      ? 'Temporarily paused; no new evidence is being generated.'
      : campaign.status === 'completed'
        ? 'Research thread completed; decisions are finalized.'
        : 'Not actively progressing.';
  const nextBestAction = experiments.length === 0
    ? 'Run a first experiment to create evidence for this objective.'
    : hypotheses.length === 0
      ? 'Capture at least one explicit hypothesis from current evidence.'
      : decisions.length === 0
        ? 'Record a go/no-go decision from the strongest evidence.'
        : 'Generate a campaign report to summarize what changed and why.';
  const sectionTabs = [
    { key: 'timeline', label: `Timeline (${experiments.length})` },
    { key: 'hypotheses', label: `Hypotheses (${hypotheses.length})` },
    { key: 'decisions', label: `Decisions (${decisions.length})` },
    { key: 'report', label: 'Report' },
  ];

  return (
    <div>
      <button onClick={onBack} className="refresh-btn" style={{ marginBottom: 16, fontSize: 12 }}>
        &larr; Back to Campaigns
      </button>

      {/* Header */}
      <div className="card" style={{ padding: 16, marginBottom: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 style={{ margin: 0, fontSize: 18 }}>{campaign.title}</h2>
          <StatusBadge status={campaign.status} colors={STATUS_COLORS} />
        </div>
        <p style={{ fontSize: 13, color: 'var(--text-secondary)', margin: '8px 0 4px' }}>
          <strong>Objective:</strong> {campaign.objective}
        </p>
        <p style={{ fontSize: 13, color: 'var(--text-secondary)', margin: '4px 0' }}>
          <strong>Success Criteria:</strong> {campaign.success_criteria}
        </p>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: '4px 0 0' }}>
          Campaign detail links objective, experiment evidence, hypothesis outcomes, and decisions in one place.
        </p>
        <p style={{ fontSize: 11, color: 'var(--text-muted)', margin: '8px 0 0' }}>
          Last updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : 'loading'} · Source: /api/campaigns/{campaignId}
        </p>

        {/* Stats */}
        <div style={{ display: 'flex', gap: 24, marginTop: 12, fontSize: 13 }}>
          <div><span style={{ color: 'var(--text-muted)' }}>Experiments:</span> {experiments.length}</div>
          <div><span style={{ color: 'var(--text-muted)' }}>Hypotheses:</span> {hypotheses.length}</div>
          <div><span style={{ color: 'var(--accent-green)' }}>Confirmed:</span> {confirmed}</div>
          <div><span style={{ color: 'var(--accent-red)' }}>Refuted:</span> {refuted}</div>
          <div><span style={{ color: 'var(--text-muted)' }}>Decisions:</span> {decisions.length}</div>
        </div>

        <div style={{ marginTop: 12 }}>
          <button
            className="start-btn"
            onClick={generateReport}
            disabled={generating}
            style={{ padding: '6px 16px', fontSize: 12 }}
          >
            {generating ? 'Generating...' : 'Generate Report'}
          </button>
        </div>
      </div>

      <div className="card" style={{ padding: 16, marginBottom: 16 }}>
        <h3 style={{ fontSize: 14, marginBottom: 10 }}>Campaign Purpose & Status</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 8 }}>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
            <strong>Why this campaign exists:</strong> {campaign.objective}
          </div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
            <strong>Current status:</strong> {statusMeaning}
          </div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
            <strong>Current evidence state:</strong> {confirmed} confirmed, {refuted} refuted, {pendingHypotheses} pending hypotheses.
          </div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
            <strong>Next best action:</strong> {nextBestAction}
          </div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
            <strong>Progress health:</strong>{' '}
            <span style={{ color: campaignHealth.color, fontWeight: 600 }}>{campaignHealth.label}</span>
          </div>
          <div style={{ marginTop: 4, display: 'grid', gap: 8 }}>
            {progressSignals.map(signal => (
              <div key={signal.label}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                  <span style={{ color: 'var(--text-secondary)' }}>{signal.label}</span>
                  <span style={{ color: 'var(--text-muted)' }}>{signal.pct}%</span>
                </div>
                <div style={{ height: 8, borderRadius: 999, background: 'var(--bg-tertiary)', overflow: 'hidden' }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${signal.pct}%`,
                      background: signal.pct >= 70 ? 'var(--accent-green)' : signal.pct >= 40 ? 'var(--accent-yellow)' : 'var(--accent-red)',
                    }}
                  />
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>{signal.detail}</div>
              </div>
            ))}
          </div>
          {blockers.length > 0 && (
            <div style={{ marginTop: 4, padding: '8px 10px', borderRadius: 6, background: 'var(--bg-tertiary)' }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
                Current blockers
              </div>
              <ul style={{ margin: 0, paddingLeft: 16, color: 'var(--text-secondary)', fontSize: 12, lineHeight: 1.5 }}>
                {blockers.map((item, idx) => (
                  <li key={`${item}-${idx}`}>{item}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* Section tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        {sectionTabs.map(tab => (
          <button
            key={tab.key}
            className={`tab ${activeSection === tab.key ? 'active' : ''}`}
            onClick={() => setActiveSection(tab.key)}
            style={{ padding: '4px 12px', fontSize: 12 }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Timeline */}
      {activeSection === 'timeline' && (
        <div className="card" style={{ padding: 16 }}>
          <h3 style={{ fontSize: 14, marginBottom: 12 }}>Experiment Timeline</h3>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
            This timeline is the evidence trail for the campaign. Open any experiment to inspect exact programs,
            failures, and metrics that informed the hypotheses and decisions.
          </p>
          {experiments.length === 0 ? (
            <p style={{ color: 'var(--text-muted)' }}>No experiments yet.</p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {experiments.map((exp, i) => {
                const linkedHyp = hypotheses.find(h => h.experiment_id === exp.experiment_id);
                return (
                  <div key={exp.experiment_id} style={{
                    padding: 10, background: 'var(--bg-secondary)', borderRadius: 4,
                    fontSize: 13,
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>
                        <strong>#{i + 1}</strong> [{exp.experiment_type}] {exp.experiment_id.slice(0, 8)}
                      </span>
                      <span style={{ color: 'var(--text-muted)' }}>
                        {exp.n_stage1_passed || 0}/{exp.n_programs_generated || 0} S1
                      </span>
                    </div>
                    {exp.hypothesis && (
                      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 4 }}>
                        {exp.hypothesis.slice(0, 100)}
                      </div>
                    )}
                    {linkedHyp && (
                      <div style={{ marginTop: 4 }}>
                        <StatusBadge status={linkedHyp.status} colors={HYPOTHESIS_COLORS} />
                      </div>
                    )}
                    {onSelectExperiment && exp.experiment_id && (
                      <div style={{ marginTop: 8, display: 'flex', justifyContent: 'flex-end' }}>
                        <button
                          className="refresh-btn"
                          style={{ fontSize: 11, padding: '4px 10px' }}
                          onClick={() => onSelectExperiment(exp.experiment_id)}
                          aria-label={`Open experiment ${exp.experiment_id}`}
                        >
                          Open Experiment
                        </button>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Hypotheses */}
      {activeSection === 'hypotheses' && (
        <div className="card" style={{ padding: 16 }}>
          <h3 style={{ fontSize: 14, marginBottom: 12 }}>Hypothesis Chain</h3>
          {hypotheses.length === 0 ? (
            <p style={{ color: 'var(--text-muted)' }}>
              No hypotheses are logged yet for this campaign.
            </p>
          ) : (
            <HypothesisChain hypotheses={hypotheses} />
          )}
        </div>
      )}

      {/* Decisions */}
      {activeSection === 'decisions' && (
        <div className="card" style={{ padding: 16 }}>
          <h3 style={{ fontSize: 14, marginBottom: 12 }}>Decision Log</h3>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
            Each decision shows traceability links to its supporting hypothesis and experiment evidence where IDs were recorded.
          </p>
          {decisions.length === 0 ? (
            <p style={{ color: 'var(--text-muted)' }}>
              No go/no-go decisions have been recorded yet.
            </p>
          ) : (
            <DecisionLog
              decisions={decisions}
              hypotheses={hypotheses}
              experiments={experiments}
              onSelectExperiment={onSelectExperiment}
            />
          )}
        </div>
      )}

      {/* Report */}
      {activeSection === 'report' && (
        <div className="card" style={{ padding: 16 }}>
          <h3 style={{ fontSize: 14, marginBottom: 12 }}>Campaign Report</h3>
          {reportError && (
            <p style={{ color: 'var(--accent-red)', marginBottom: 8 }}>{reportError}</p>
          )}
          {report ? (
            <>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
                Report generated: {reportGeneratedAt ? reportGeneratedAt.toLocaleTimeString() : 'just now'} · Source: /api/campaigns/{campaignId}/report
              </div>
              <div style={{ whiteSpace: 'pre-wrap', fontSize: 13, lineHeight: 1.6, color: 'var(--text-secondary)' }}>
                {report.report}
              </div>
            </>
          ) : !reportError && (
            <p style={{ color: 'var(--text-muted)' }}>
              Click "Generate Report" above to compile the current hypotheses, evidence, and decisions into one narrative.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function CampaignView({ onSelectExperiment }) {
  const [selectedCampaign, setSelectedCampaign] = useState(null);

  if (selectedCampaign) {
    return (
      <CampaignDetail
        campaignId={selectedCampaign}
        onBack={() => setSelectedCampaign(null)}
        onSelectExperiment={onSelectExperiment}
      />
    );
  }

  return (
    <div>
      <h2 style={{ fontSize: 16, marginBottom: 16 }}>Research Campaigns</h2>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Campaigns are long-running research threads: they connect one objective to the experiments run, hypotheses tested,
        and decisions made. Use Open Details to inspect the evidence trail and generate a campaign narrative.
      </p>
      <CampaignList onSelectCampaign={setSelectedCampaign} />
    </div>
  );
}

export default CampaignView;
