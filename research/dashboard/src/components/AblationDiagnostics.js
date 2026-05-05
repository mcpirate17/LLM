import React, { useEffect, useMemo, useState } from 'react';
import { apiService } from '../services/apiService';

const fmtInt = (v) => Number(v || 0).toLocaleString();
const fmtPct = (v) => `${((Number(v || 0)) * 100).toFixed(1)}%`;
const fmtSigned = (v, digits = 3) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return '--';
  const sign = n > 0 ? '+' : '';
  return `${sign}${n.toFixed(digits)}`;
};
const fmtNum = (v, digits = 3) => {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(digits) : '--';
};
const shortFp = (fp) => (fp ? String(fp).slice(0, 14) : '');
const shortRid = (rid) => (rid ? String(rid).slice(0, 12) : '');

// ── Recommendation classifier ────────────────────────────────────────────
// "Loss alone is not enough" — judge a rule on a basket of metrics.
// Each metric Δ is "ablation child − parent" along the helpful direction.
// Positive Δ ⇒ removing/changing the component HURT, so the component is USEFUL.
// Negative Δ ⇒ removing/changing the component HELPED, so the component is BAGGAGE.
const METRIC_AXES = [
  { key: 'avg_d_induction_v2', n: 'n_induction_v2', label: 'Δ ind v2', helpful: 'pos' },
  { key: 'avg_d_binding_v2',   n: 'n_binding_v2',   label: 'Δ bind v2', helpful: 'pos' },
  { key: 'avg_d_induction', n: 'n_induction', label: 'Δ ind', helpful: 'pos' },
  { key: 'avg_d_binding',   n: 'n_binding',   label: 'Δ bind', helpful: 'pos' },
  { key: 'avg_d_ar',        n: 'n_ar',        label: 'Δ ar',   helpful: 'pos' },
  { key: 'avg_d_blimp',     n: 'n_blimp',     label: 'Δ blimp', helpful: 'pos' },
  { key: 'avg_d_hellaswag', n: 'n_hellaswag', label: 'Δ hsw',  helpful: 'pos' },
  { key: 'avg_d_ppl_pct',   n: 'n_ppl',       label: 'Δ ppl%', helpful: 'pos' },
  { key: 'avg_d_loss',      n: 'n_loss',      label: 'Δ loss', helpful: 'pos' },
];

function classifyRecommendation(row) {
  const supports = [];
  const refutes = [];
  for (const ax of METRIC_AXES) {
    const v = Number(row[ax.key]);
    const n = Number(row[ax.n] || 0);
    if (!Number.isFinite(v) || n < 2) continue;
    if (v >= 0.005) supports.push(ax.label);
    else if (v <= -0.005) refutes.push(ax.label);
  }
  if (supports.length >= 2 && refutes.length === 0) return { tag: 'use', supports, refutes };
  if (refutes.length >= 2 && supports.length === 0) return { tag: 'avoid', supports, refutes };
  if (supports.length || refutes.length) return { tag: 'mixed', supports, refutes };
  return { tag: 'inconclusive', supports, refutes };
}

const TAG_COLORS = {
  use: 'var(--accent-green)',
  avoid: 'var(--accent-red)',
  mixed: 'var(--accent-yellow)',
  inconclusive: 'var(--text-muted)',
};
const TAG_GLYPH = {
  use: '✓ USE',
  avoid: '✗ AVOID',
  mixed: '⚠ MIXED',
  inconclusive: '· n/a',
};

// ── Top-level layout pieces ──────────────────────────────────────────────
function Banner({ totals }) {
  const backfillGap = totals.backfill_gap || {};
  const recent = totals.recent_24h || {};
  const missing = Number(backfillGap.s1_missing_core_metrics || 0);
  const s1Rows = Number(backfillGap.s1_ablation_rows || 0);
  const coverageOk = s1Rows > 0 && missing === 0;
  return (
    <div className="card" style={{ padding: 16, display: 'grid', gap: 8 }}>
      <div className="card-title" style={{ marginBottom: 0 }}>Causal Ablation Diagnostics</div>
      <div style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.5 }}>
        Children built by mutating leaderboard parents. Loss alone is not enough — the
        page judges every rule on induction, binding, AR, BLiMP, HellaSwag and WikiText
        perplexity. Loss-only rows show as "incomplete" rather than known-good/known-bad.
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, fontSize: 12 }}>
        <span><strong>{fmtInt(s1Rows)}</strong> S1 ablation rows</span>
        <span style={{ color: coverageOk ? 'var(--accent-green)' : 'var(--accent-red)' }}>
          <strong>{fmtInt(missing)}</strong> missing core metrics{coverageOk ? ' ✓' : ''}
        </span>
        <span><strong>{fmtInt(totals.evidence_count)}</strong> rules</span>
        <span><strong>{fmtInt(totals.observation_count)}</strong> observations</span>
        <span style={{ color: 'var(--text-muted)' }}>·</span>
        <span>last 24h: <strong>{fmtInt(recent.evidence_count)}</strong> rules
          (<span style={{ color: 'var(--accent-green)' }}>{fmtInt(recent.supported_count)}</span>
          {' / '}
          <span style={{ color: 'var(--accent-red)' }}>{fmtInt(recent.refuted_count)}</span>)
        </span>
      </div>
    </div>
  );
}

function ViewSwitcher({ value, onChange }) {
  const tabs = [
    { id: 'recs',       label: 'Recommendations' },
    { id: 'champions',  label: 'By Champion' },
    { id: 'components', label: 'By Component' },
    { id: 'rules',      label: 'All Rules' },
  ];
  return (
    <div className="card" style={{ padding: 6, display: 'flex', gap: 4 }}>
      {tabs.map((t) => (
        <button
          key={t.id}
          className={value === t.id ? 'tab active' : 'tab'}
          onClick={() => onChange(t.id)}
          style={{
            padding: '6px 14px',
            border: 'none',
            borderRadius: 4,
            cursor: 'pointer',
            fontSize: 12,
            fontWeight: 600,
            background: value === t.id ? 'var(--accent-blue)' : 'transparent',
            color: value === t.id ? 'white' : 'var(--text-secondary)',
          }}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

// ── Active construction prior bar ───────────────────────────────────────
function ConstructionPriorBar() {
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState('');

  const reload = () => {
    apiService.getConstructionPrior()
      .then((d) => setData(d || {}))
      .catch((e) => setMsg(e.message || 'load failed'));
  };
  useEffect(reload, []);

  const refresh = async () => {
    setBusy(true); setMsg('');
    try {
      const res = await apiService.refreshConstructionPrior({ min_n: 4 });
      setMsg(`Activated ${res.version}: ${res.summary?.n_use || 0} use / ${res.summary?.n_avoid || 0} avoid / ${res.summary?.n_mixed || 0} mixed`);
      reload();
    } catch (e) {
      setMsg(`Error: ${e.message || e}`);
    } finally {
      setBusy(false);
    }
  };

  const active = data?.active;
  const summary = active?.summary || {};
  return (
    <div className="card" style={{ padding: 12, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
      <div style={{ fontSize: 12 }}>
        <strong style={{ color: 'var(--accent-blue)' }}>Active Construction Prior:</strong>
        {' '}
        {active ? (
          <>
            <code style={{ background: 'var(--bg-tertiary)', padding: '1px 6px', borderRadius: 3 }}>{active.version}</code>
            {' · '}
            <span style={{ color: 'var(--accent-green)' }}>{summary.n_use || 0} use</span>
            {' / '}
            <span style={{ color: 'var(--accent-red)' }}>{summary.n_avoid || 0} avoid</span>
            {' / '}
            <span style={{ color: 'var(--text-muted)' }}>{summary.n_mixed || 0} mixed</span>
            {' · '}
            <span style={{ color: 'var(--text-muted)' }}>op_w={summary.n_op_weights || 0}, slot_motif={summary.n_slot_motif_multipliers || 0}, deny={summary.n_slot_motif_denylist || 0}</span>
          </>
        ) : <span style={{ color: 'var(--text-muted)' }}>none — grammar uses defaults</span>}
        {msg && <div style={{ marginTop: 4, fontSize: 11, color: msg.startsWith('Error') ? 'var(--accent-red)' : 'var(--accent-green)' }}>{msg}</div>}
      </div>
      <button
        onClick={refresh}
        disabled={busy}
        style={{
          padding: '6px 14px', fontSize: 12, fontWeight: 600,
          background: 'var(--accent-blue)', color: 'white',
          border: 'none', borderRadius: 4, cursor: busy ? 'wait' : 'pointer',
        }}
        title="Recompute prior from current evidence and activate as new snapshot. The grammar will use it on the next screening run."
      >
        {busy ? 'Computing…' : 'Refresh & Activate'}
      </button>
    </div>
  );
}

// ── Recommendations view ─────────────────────────────────────────────────
function RecommendationsView({ onDrill }) {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [filter, setFilter] = useState('all');
  const [minN, setMinN] = useState(4);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    apiService.getCausalAblationRecommendations({ min_n: minN, limit: 200 })
      .then((data) => {
        if (cancelled) return;
        const cls = (data?.recommendations || []).map((r) => ({
          ...r,
          _rec: classifyRecommendation(r),
        }));
        setRows(cls);
        setError('');
      })
      .catch((err) => { if (!cancelled) setError(err.message || 'load failed'); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [minN]);

  const filtered = useMemo(() => {
    if (filter === 'all') return rows;
    return rows.filter((r) => r._rec.tag === filter);
  }, [rows, filter]);

  const byTag = useMemo(() => ({
    use: rows.filter((r) => r._rec.tag === 'use').length,
    avoid: rows.filter((r) => r._rec.tag === 'avoid').length,
    mixed: rows.filter((r) => r._rec.tag === 'mixed').length,
    inconclusive: rows.filter((r) => r._rec.tag === 'inconclusive').length,
  }), [rows]);

  if (loading) return <div className="ux-state ux-state-loading">Loading recommendations...</div>;
  if (error)   return <div className="ux-state ux-state-error">{error}</div>;

  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <div className="card-title" style={{ margin: 0 }}>Construction Recommendations</div>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 11 }}>
          <span style={{ color: 'var(--text-muted)' }}>min n:</span>
          <select value={minN} onChange={(e) => setMinN(Number(e.target.value))} style={{ fontSize: 11 }}>
            {[2, 3, 4, 6, 10].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
      </div>
      <div style={{ display: 'flex', gap: 6, marginBottom: 12, flexWrap: 'wrap' }}>
        {[
          ['all', `All (${rows.length})`, 'var(--text-secondary)'],
          ['use', `✓ USE (${byTag.use})`, TAG_COLORS.use],
          ['avoid', `✗ AVOID (${byTag.avoid})`, TAG_COLORS.avoid],
          ['mixed', `⚠ MIXED (${byTag.mixed})`, TAG_COLORS.mixed],
          ['inconclusive', `· inconclusive (${byTag.inconclusive})`, TAG_COLORS.inconclusive],
        ].map(([id, label, color]) => (
          <button
            key={id}
            onClick={() => setFilter(id)}
            style={{
              padding: '4px 10px',
              borderRadius: 12,
              border: filter === id ? `1px solid ${color}` : '1px solid var(--border-color)',
              background: filter === id ? `${color}22` : 'transparent',
              color, fontSize: 11, fontWeight: 600, cursor: 'pointer',
            }}
          >{label}</button>
        ))}
      </div>
      <div className="table-scroll">
        <table className="data-table compact">
          <thead>
            <tr>
              <th style={{ width: 90 }}>Verdict</th>
              <th>Rule</th>
              <th>n</th>
              <th>Contexts</th>
              <th>Δ ind v2</th>
              <th>Δ bind v2</th>
              <th>Δ ind</th>
              <th>Δ bind</th>
              <th>Δ ar</th>
              <th>Δ blimp</th>
              <th>Δ hsw</th>
              <th>Δ ppl%</th>
              <th>Δ loss</th>
              <th>Coverage</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((row) => {
              const rec = row._rec;
              const color = TAG_COLORS[rec.tag];
              const n = Number(row.n || 0);
              const cov = n > 0 ? Number(row.metric_complete_count || 0) / n : 0;
              return (
                <tr
                  key={`${row.rule_type}:${row.rule_key}`}
                  onClick={() => onDrill({ rule_type: row.rule_type, rule_key: row.rule_key })}
                  style={{ cursor: 'pointer' }}
                >
                  <td style={{ color, fontWeight: 700 }}>{TAG_GLYPH[rec.tag]}</td>
                  <td style={{ maxWidth: 360 }}>
                    <div style={{ fontWeight: 600 }}>{row.rule_key}</div>
                    <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>{row.rule_type}</div>
                  </td>
                  <td>{fmtInt(n)}</td>
                  <td>{fmtInt(row.contexts)}</td>
                  <td style={{ color: signColor(row.avg_d_induction_v2) }}>{fmtSigned(row.avg_d_induction_v2)}</td>
                  <td style={{ color: signColor(row.avg_d_binding_v2) }}>{fmtSigned(row.avg_d_binding_v2)}</td>
                  <td style={{ color: signColor(row.avg_d_induction) }}>{fmtSigned(row.avg_d_induction)}</td>
                  <td style={{ color: signColor(row.avg_d_binding) }}>{fmtSigned(row.avg_d_binding)}</td>
                  <td style={{ color: signColor(row.avg_d_ar) }}>{fmtSigned(row.avg_d_ar)}</td>
                  <td style={{ color: signColor(row.avg_d_blimp) }}>{fmtSigned(row.avg_d_blimp)}</td>
                  <td style={{ color: signColor(row.avg_d_hellaswag) }}>{fmtSigned(row.avg_d_hellaswag)}</td>
                  <td style={{ color: signColor(row.avg_d_ppl_pct) }}>{fmtSigned(row.avg_d_ppl_pct, 2)}</td>
                  <td style={{ color: signColor(row.avg_d_loss) }}>{fmtSigned(row.avg_d_loss, 4)}</td>
                  <td>{fmtPct(cov)}</td>
                </tr>
              );
            })}
            {filtered.length === 0 && (
              <tr><td colSpan="14" style={{ textAlign: 'center', padding: 18, color: 'var(--text-muted)' }}>
                No recommendations at this filter / min n.
              </td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function signColor(v) {
  const n = Number(v);
  if (!Number.isFinite(n) || Math.abs(n) < 1e-6) return 'var(--text-muted)';
  return n > 0 ? 'var(--accent-green)' : 'var(--accent-red)';
}

// ── Champion view ────────────────────────────────────────────────────────
function ChampionsView({ onDrill }) {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    apiService.getCausalAblationChampions(80)
      .then((data) => { if (!cancelled) { setRows(data?.champions || []); setError(''); } })
      .catch((err) => { if (!cancelled) setError(err.message || 'load failed'); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  if (loading) return <div className="ux-state ux-state-loading">Loading champions...</div>;
  if (error)   return <div className="ux-state ux-state-error">{error}</div>;

  return (
    <div className="card" style={{ padding: 16 }}>
      <div className="card-title">Per-Champion Ablation Rollup</div>
      <div style={{ color: 'var(--text-muted)', fontSize: 11, marginTop: 4, marginBottom: 8 }}>
        Click a row to see all rules tested against that champion. "Drop" columns show
        mean child−parent change along the metric's helpful direction (positive = ablation hurt,
        the component is pulling its weight).
      </div>
      <div className="table-scroll">
        <table className="data-table compact">
          <thead>
            <tr>
              <th>Champion (rid · fp)</th>
              <th>Score</th>
              <th>Tier</th>
              <th>Children</th>
              <th>Coverage</th>
              <th>Sup / Ref</th>
              <th>Mean Δ ind v2</th>
              <th>Mean Δ bind v2</th>
              <th>Mean Δ ind</th>
              <th>Mean Δ bind</th>
              <th>Mean Δ ar</th>
              <th>Mean Δ blimp</th>
              <th>Mean Δ hsw</th>
              <th>Mean Δ ppl%</th>
              <th>Mean Δ loss</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const cov = Number(row.metric_complete_rate || 0);
              return (
                <tr
                  key={row.result_id}
                  onClick={() => onDrill({ parent_result_id: row.result_id })}
                  style={{ cursor: 'pointer' }}
                >
                  <td>
                    <div style={{ fontWeight: 600 }}>{shortRid(row.result_id)}</div>
                    <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>{shortFp(row.graph_fingerprint)}</div>
                  </td>
                  <td>{fmtNum(row.composite_score, 1)}</td>
                  <td><span className={`tier-badge tier-${(row.tier || '').toLowerCase()}`}>{row.tier || '—'}</span></td>
                  <td>
                    {fmtInt(row.evidence_count)}
                    <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>
                      {fmtInt(row.child_fingerprint_count)} unique fp
                    </div>
                  </td>
                  <td style={{ color: cov >= 0.8 ? 'var(--accent-green)' : 'var(--accent-yellow)' }}>
                    {fmtPct(cov)}
                  </td>
                  <td>
                    <span style={{ color: 'var(--accent-green)' }}>{fmtInt(row.supported_count)}</span>
                    {' / '}
                    <span style={{ color: 'var(--accent-red)' }}>{fmtInt(row.refuted_count)}</span>
                  </td>
                  <td style={{ color: signColor(row.avg_induction_v2_drop) }}>
                    {fmtSigned(row.avg_induction_v2_drop)}
                    <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>{fmtInt(row.induction_v2_count)} v2</div>
                  </td>
                  <td style={{ color: signColor(row.avg_binding_v2_drop) }}>
                    {fmtSigned(row.avg_binding_v2_drop)}
                    <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>{fmtInt(row.binding_v2_count)} v2</div>
                  </td>
                  <td style={{ color: signColor(row.avg_induction_drop) }}>{fmtSigned(row.avg_induction_drop)}</td>
                  <td style={{ color: signColor(row.avg_binding_drop) }}>{fmtSigned(row.avg_binding_drop)}</td>
                  <td style={{ color: signColor(row.avg_ar_drop) }}>{fmtSigned(row.avg_ar_drop)}</td>
                  <td style={{ color: signColor(row.avg_blimp_drop) }}>{fmtSigned(row.avg_blimp_drop)}</td>
                  <td style={{ color: signColor(row.avg_hellaswag_drop) }}>{fmtSigned(row.avg_hellaswag_drop)}</td>
                  <td style={{ color: signColor(row.avg_ppl_pct_change) }}>{fmtSigned(row.avg_ppl_pct_change, 2)}</td>
                  <td style={{ color: signColor(row.avg_loss_delta) }}>{fmtSigned(row.avg_loss_delta, 4)}</td>
                </tr>
              );
            })}
            {rows.length === 0 && (
              <tr><td colSpan="15" style={{ textAlign: 'center', padding: 18, color: 'var(--text-muted)' }}>No champions with ablation evidence yet.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Component view ───────────────────────────────────────────────────────
function ComponentsView({ onDrill }) {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [ruleType, setRuleType] = useState('');

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    const params = ruleType ? { rule_type: ruleType, limit: 300 } : { limit: 300 };
    apiService.getCausalAblationComponents(params)
      .then((data) => { if (!cancelled) { setRows(data?.components || []); setError(''); } })
      .catch((err) => { if (!cancelled) setError(err.message || 'load failed'); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [ruleType]);

  if (loading) return <div className="ux-state ux-state-loading">Loading components...</div>;
  if (error)   return <div className="ux-state ux-state-error">{error}</div>;

  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <div className="card-title" style={{ margin: 0 }}>By Component</div>
        <div style={{ display: 'flex', gap: 4 }}>
          {[
            '',
            'node_delete_investigation',
            'node_delete_s1',
            'node_delete',
            'component_replace',
            'op_pair',
            'slot_motif',
            'op',
          ].map((t) => (
            <button
              key={t || 'all'}
              onClick={() => setRuleType(t)}
              style={{
                padding: '4px 10px', fontSize: 11, fontWeight: 600,
                background: ruleType === t ? 'var(--accent-blue)' : 'transparent',
                color: ruleType === t ? 'white' : 'var(--text-secondary)',
                border: '1px solid var(--border-color)', borderRadius: 12, cursor: 'pointer',
              }}
            >{t || 'all'}</button>
          ))}
        </div>
      </div>
      <div className="table-scroll">
        <table className="data-table compact">
          <thead>
            <tr>
              <th>Rule</th>
              <th>n</th>
              <th>Contexts</th>
              <th>Δ ind v2 (n)</th>
              <th>Δ bind v2 (n)</th>
              <th>Δ ind (n)</th>
              <th>Δ bind (n)</th>
              <th>Δ ar (n)</th>
              <th>Δ blimp (n)</th>
              <th>Δ hsw (n)</th>
              <th>Δ ppl% (n)</th>
              <th>Δ loss (n)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={`${row.rule_type}:${row.rule_key}`}
                onClick={() => onDrill({ rule_type: row.rule_type, rule_key: row.rule_key })}
                style={{ cursor: 'pointer' }}
              >
                <td style={{ maxWidth: 360 }}>
                  <div style={{ fontWeight: 600 }}>{row.rule_key}</div>
                  <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>{row.rule_type}</div>
                </td>
                <td>{fmtInt(row.observation_count)}</td>
                <td>{fmtInt(row.parent_count)}</td>
                <td style={{ color: signColor(row.avg_d_induction_v2) }}>
                  {fmtSigned(row.avg_d_induction_v2)} <span style={{ color: 'var(--text-muted)' }}>({fmtInt(row.n_induction_v2)})</span>
                </td>
                <td style={{ color: signColor(row.avg_d_binding_v2) }}>
                  {fmtSigned(row.avg_d_binding_v2)} <span style={{ color: 'var(--text-muted)' }}>({fmtInt(row.n_binding_v2)})</span>
                </td>
                <td style={{ color: signColor(row.avg_d_induction) }}>
                  {fmtSigned(row.avg_d_induction)} <span style={{ color: 'var(--text-muted)' }}>({fmtInt(row.n_induction)})</span>
                </td>
                <td style={{ color: signColor(row.avg_d_binding) }}>
                  {fmtSigned(row.avg_d_binding)} <span style={{ color: 'var(--text-muted)' }}>({fmtInt(row.n_binding)})</span>
                </td>
                <td style={{ color: signColor(row.avg_d_ar) }}>
                  {fmtSigned(row.avg_d_ar)} <span style={{ color: 'var(--text-muted)' }}>({fmtInt(row.n_ar)})</span>
                </td>
                <td style={{ color: signColor(row.avg_d_blimp) }}>
                  {fmtSigned(row.avg_d_blimp)} <span style={{ color: 'var(--text-muted)' }}>({fmtInt(row.n_blimp)})</span>
                </td>
                <td style={{ color: signColor(row.avg_d_hellaswag) }}>
                  {fmtSigned(row.avg_d_hellaswag)} <span style={{ color: 'var(--text-muted)' }}>({fmtInt(row.n_hellaswag)})</span>
                </td>
                <td style={{ color: signColor(row.avg_d_ppl_pct) }}>
                  {fmtSigned(row.avg_d_ppl_pct, 2)} <span style={{ color: 'var(--text-muted)' }}>({fmtInt(row.n_ppl)})</span>
                </td>
                <td style={{ color: signColor(row.avg_d_loss) }}>
                  {fmtSigned(row.avg_d_loss, 4)} <span style={{ color: 'var(--text-muted)' }}>({fmtInt(row.n_loss)})</span>
                </td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr><td colSpan="12" style={{ textAlign: 'center', padding: 18, color: 'var(--text-muted)' }}>No components.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── All-rules legacy view (compact table) ────────────────────────────────
function AllRulesView({ summary, onDrill }) {
  const [filter, setFilter] = useState('all');
  const [search, setSearch] = useState('');
  const filtered = useMemo(() => {
    let r = summary;
    if (filter === 'credible') {
      r = r.filter((row) =>
        Number(row.evidence_count || 0) >= 3
        && Number(row.child_fingerprint_count || 0) >= 3
        && Number(row.metric_complete_count || 0) >= 3
        && Number(row.metric_complete_rate || 0) >= 0.8);
    } else if (filter === 'incomplete') {
      r = r.filter((row) => Number(row.metric_complete_rate || 0) < 0.8);
    }
    if (search) {
      const q = search.toLowerCase();
      r = r.filter((row) =>
        String(row.rule_key || '').toLowerCase().includes(q)
        || String(row.rule_type || '').toLowerCase().includes(q));
    }
    return r;
  }, [summary, filter, search]);

  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <div className="card-title" style={{ margin: 0 }}>All Rules</div>
        <input
          placeholder="search rule_key…" value={search} onChange={(e) => setSearch(e.target.value)}
          style={{ fontSize: 11, padding: '4px 8px', width: 200 }}
        />
      </div>
      <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
        {['all', 'credible', 'incomplete'].map((id) => (
          <button
            key={id} onClick={() => setFilter(id)}
            style={{
              padding: '4px 10px', borderRadius: 12, border: '1px solid var(--border-color)',
              background: filter === id ? 'var(--accent-blue)' : 'transparent',
              color: filter === id ? 'white' : 'var(--text-secondary)',
              fontSize: 11, fontWeight: 600, cursor: 'pointer',
            }}
          >{id}</button>
        ))}
      </div>
      <div className="table-scroll">
        <table className="data-table compact">
          <thead>
            <tr>
              <th>Rule</th>
              <th>Evidence</th>
              <th>Fingerprints</th>
              <th>Sup / Ref</th>
              <th>Composite Δ</th>
              <th>Δ ind v2</th>
              <th>Δ bind v2</th>
              <th>Δ ind</th>
              <th>Δ bind</th>
              <th>Δ ar</th>
              <th>Coverage</th>
            </tr>
          </thead>
          <tbody>
            {filtered.slice(0, 80).map((row) => (
              <tr
                key={`${row.rule_type}:${row.rule_key}`}
                onClick={() => onDrill({ rule_type: row.rule_type, rule_key: row.rule_key })}
                style={{ cursor: 'pointer' }}
              >
                <td style={{ maxWidth: 360 }}>
                  <div style={{ fontWeight: 600 }}>{row.rule_key}</div>
                  <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>{row.rule_type}</div>
                </td>
                <td>{fmtInt(row.evidence_count)}</td>
                <td>{fmtInt(row.child_fingerprint_count)}</td>
                <td>
                  <span style={{ color: 'var(--accent-green)' }}>{fmtInt(row.supported_count)}</span>
                  {' / '}
                  <span style={{ color: 'var(--accent-red)' }}>{fmtInt(row.refuted_count)}</span>
                </td>
                <td style={{ color: signColor(row.composite_support_effect) }}>{fmtSigned(row.composite_support_effect)}</td>
                <td style={{ color: signColor(row.avg_induction_v2_support_effect) }}>{fmtSigned(row.avg_induction_v2_support_effect)}</td>
                <td style={{ color: signColor(row.avg_binding_v2_support_effect) }}>{fmtSigned(row.avg_binding_v2_support_effect)}</td>
                <td style={{ color: signColor(row.avg_induction_support_effect) }}>{fmtSigned(row.avg_induction_support_effect)}</td>
                <td style={{ color: signColor(row.avg_binding_support_effect) }}>{fmtSigned(row.avg_binding_support_effect)}</td>
                <td style={{ color: signColor(row.avg_ar_support_effect) }}>{fmtSigned(row.avg_ar_support_effect)}</td>
                <td>{fmtPct(row.metric_complete_rate)}</td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr><td colSpan="11" style={{ textAlign: 'center', padding: 18, color: 'var(--text-muted)' }}>No rows.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Drill-down drawer ────────────────────────────────────────────────────
function DrillDrawer({ params, onClose }) {
  const [children, setChildren] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!params) return;
    let cancelled = false;
    setLoading(true);
    apiService.getCausalAblationChildrenForRule(params)
      .then((data) => { if (!cancelled) { setChildren(data?.children || []); setError(''); } })
      .catch((err) => { if (!cancelled) setError(err.message || 'load failed'); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [params]);

  if (!params) return null;

  return (
    <div
      style={{
        position: 'fixed', top: 0, right: 0, bottom: 0, width: '60vw', maxWidth: 1100,
        background: 'var(--bg-primary)', borderLeft: '1px solid var(--border-color)',
        boxShadow: '-4px 0 16px rgba(0,0,0,0.3)', zIndex: 100, overflowY: 'auto',
      }}
    >
      <div style={{ padding: 16, borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between' }}>
        <div>
          <div className="card-title" style={{ margin: 0 }}>
            {params.rule_type ? `${params.rule_type}: ${params.rule_key}` : `Champion ${shortRid(params.parent_result_id)}`}
          </div>
          <div style={{ color: 'var(--text-muted)', fontSize: 11, marginTop: 2 }}>
            Child observations with full per-metric numbers (parent → child).
          </div>
        </div>
        <button onClick={onClose} style={{
          background: 'transparent', border: '1px solid var(--border-color)', color: 'var(--text-secondary)',
          padding: '4px 12px', cursor: 'pointer', borderRadius: 4, fontSize: 12,
        }}>close</button>
      </div>
      <div style={{ padding: 16 }}>
        {loading && <div className="ux-state ux-state-loading">Loading...</div>}
        {error && <div className="ux-state ux-state-error">{error}</div>}
        {!loading && !error && (
          <div className="table-scroll">
            <table className="data-table compact">
              <thead>
                <tr>
                  <th>Child fp</th>
                  <th>Source</th>
                  <th>Loss ratio (P → C)</th>
                  <th>PPL (P → C)</th>
                  <th>Ind v2 (P → C)</th>
                  <th>Bind v2 (P → C)</th>
                  <th>Induction (P → C)</th>
                  <th>Binding (P → C)</th>
                  <th>AR (P → C)</th>
                  <th>HellaSwag (P → C)</th>
                  <th>BLiMP (P → C)</th>
                  <th>Trust</th>
                </tr>
              </thead>
              <tbody>
                {children.map((c) => (
                  <tr key={c.child_result_id}>
                    <td>
                      <div style={{ fontWeight: 600 }}>{shortFp(c.child_fingerprint)}</div>
                      <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>{shortRid(c.child_result_id)}</div>
                    </td>
                    <td><span className={`tag tag-${c.source}`}>{c.source}</span></td>
                    <td>{fmtNum(c.parent_loss_ratio, 4)} → {fmtNum(c.child_loss_ratio, 4)}</td>
                    <td>{fmtNum(c.parent_ppl, 1)} → {fmtNum(c.child_ppl, 1)}</td>
                    <td>
                      {fmtNum(c.parent_induction_v2, 3)} → {fmtNum(c.child_induction_v2, 3)}
                      <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>{c.child_induction_v2_status || '—'}</div>
                    </td>
                    <td>
                      {fmtNum(c.parent_binding_v2, 3)} → {fmtNum(c.child_binding_v2, 3)}
                      <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>{c.child_binding_v2_status || '—'}</div>
                    </td>
                    <td>{fmtNum(c.parent_induction, 3)} → {fmtNum(c.child_induction, 3)}</td>
                    <td>{fmtNum(c.parent_binding, 3)} → {fmtNum(c.child_binding, 3)}</td>
                    <td>{fmtNum(c.parent_ar, 3)} → {fmtNum(c.child_ar, 3)}</td>
                    <td>{fmtNum(c.parent_hellaswag, 3)} → {fmtNum(c.child_hellaswag, 3)}</td>
                    <td>{fmtNum(c.parent_blimp, 3)} → {fmtNum(c.child_blimp, 3)}</td>
                    <td style={{ fontSize: 10, color: 'var(--text-muted)' }}>{c.child_trust_label || '—'}</td>
                  </tr>
                ))}
                {children.length === 0 && (
                  <tr><td colSpan="12" style={{ textAlign: 'center', padding: 18, color: 'var(--text-muted)' }}>No child observations.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────
export default function AblationDiagnostics() {
  const [view, setView] = useState('recs');
  const [drill, setDrill] = useState(null);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    apiService.getCausalAblationSummary(500)
      .then((data) => { if (!cancelled) { setSummary(data || {}); setError(''); } })
      .catch((err) => { if (!cancelled) setError(err.message || 'load failed'); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  if (loading) return <div className="ux-state ux-state-loading">Loading diagnostics...</div>;
  if (error)   return <div className="ux-state ux-state-error">{error}</div>;

  const totals = summary?.totals || {};
  const rules = summary?.summary || [];

  return (
    <div style={{ display: 'grid', gap: 12, position: 'relative' }}>
      <Banner totals={totals} />
      <ConstructionPriorBar />
      <ViewSwitcher value={view} onChange={setView} />
      {view === 'recs'       && <RecommendationsView onDrill={setDrill} />}
      {view === 'champions'  && <ChampionsView onDrill={setDrill} />}
      {view === 'components' && <ComponentsView onDrill={setDrill} />}
      {view === 'rules'      && <AllRulesView summary={rules} onDrill={setDrill} />}
      <DrillDrawer params={drill} onClose={() => setDrill(null)} />
    </div>
  );
}
