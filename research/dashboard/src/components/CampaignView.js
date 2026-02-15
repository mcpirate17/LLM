import React, { useState, useEffect } from 'react';

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

  useEffect(() => {
    fetch(`${API_BASE}/api/campaigns`)
      .then(r => r.json())
      .then(d => { setCampaigns(Array.isArray(d) ? d : []); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  if (loading) return <p style={{ color: 'var(--text-muted)' }}>Loading campaigns...</p>;
  if (campaigns.length === 0) return <p style={{ color: 'var(--text-muted)' }}>No campaigns yet. Start a continuous experiment to auto-create one.</p>;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {campaigns.map(c => (
        <div
          key={c.campaign_id}
          className="card"
          style={{ cursor: 'pointer', padding: 16 }}
          onClick={() => onSelectCampaign(c.campaign_id)}
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

function DecisionLog({ decisions }) {
  if (!decisions || decisions.length === 0) return null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {decisions.map(d => (
        <div key={d.decision_id} style={{
          padding: 10, background: 'var(--bg-secondary)', borderRadius: 4,
          borderLeft: `3px solid ${DECISION_COLORS[d.decision_type] || 'var(--border)'}`,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <StatusBadge status={d.decision_type} colors={DECISION_COLORS} />
            <span style={{ fontSize: 13, fontWeight: 500 }}>{d.subject}</span>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{d.rationale}</div>
        </div>
      ))}
    </div>
  );
}

function CampaignDetail({ campaignId, onBack }) {
  const [data, setData] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [activeSection, setActiveSection] = useState('timeline');

  useEffect(() => {
    fetch(`${API_BASE}/api/campaigns/${campaignId}`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, [campaignId]);

  const generateReport = async () => {
    setGenerating(true);
    try {
      const r = await fetch(`${API_BASE}/api/campaigns/${campaignId}/report`);
      const d = await r.json();
      setReport(d);
    } catch (e) {
      console.error(e);
    }
    setGenerating(false);
  };

  if (loading) return <p style={{ color: 'var(--text-muted)' }}>Loading...</p>;
  if (!data) return <p style={{ color: 'var(--accent-red)' }}>Campaign not found</p>;

  const { campaign, experiments, hypotheses, decisions } = data;
  const confirmed = hypotheses.filter(h => h.status === 'confirmed').length;
  const refuted = hypotheses.filter(h => h.status === 'refuted').length;

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

      {/* Section tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        {['timeline', 'hypotheses', 'decisions', 'report'].map(s => (
          <button
            key={s}
            className={`tab ${activeSection === s ? 'active' : ''}`}
            onClick={() => setActiveSection(s)}
            style={{ padding: '4px 12px', fontSize: 12 }}
          >
            {s.charAt(0).toUpperCase() + s.slice(1)}
          </button>
        ))}
      </div>

      {/* Timeline */}
      {activeSection === 'timeline' && (
        <div className="card" style={{ padding: 16 }}>
          <h3 style={{ fontSize: 14, marginBottom: 12 }}>Experiment Timeline</h3>
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
          <HypothesisChain hypotheses={hypotheses} />
        </div>
      )}

      {/* Decisions */}
      {activeSection === 'decisions' && (
        <div className="card" style={{ padding: 16 }}>
          <h3 style={{ fontSize: 14, marginBottom: 12 }}>Decision Log</h3>
          <DecisionLog decisions={decisions} />
        </div>
      )}

      {/* Report */}
      {activeSection === 'report' && (
        <div className="card" style={{ padding: 16 }}>
          <h3 style={{ fontSize: 14, marginBottom: 12 }}>Campaign Report</h3>
          {report ? (
            <div style={{ whiteSpace: 'pre-wrap', fontSize: 13, lineHeight: 1.6, color: 'var(--text-secondary)' }}>
              {report.report}
            </div>
          ) : (
            <p style={{ color: 'var(--text-muted)' }}>
              Click "Generate Report" above to create a compiled research narrative.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function CampaignView() {
  const [selectedCampaign, setSelectedCampaign] = useState(null);

  if (selectedCampaign) {
    return (
      <CampaignDetail
        campaignId={selectedCampaign}
        onBack={() => setSelectedCampaign(null)}
      />
    );
  }

  return (
    <div>
      <h2 style={{ fontSize: 16, marginBottom: 16 }}>Research Campaigns</h2>
      <CampaignList onSelectCampaign={setSelectedCampaign} />
    </div>
  );
}

export default CampaignView;
