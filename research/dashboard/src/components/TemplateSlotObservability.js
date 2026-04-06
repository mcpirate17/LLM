import React, { useMemo, useState } from 'react';
import { useAriaData } from '../hooks/useAriaData';
import { fmtLoss, fmtNumber, fmtPct } from '../utils/format';
import useInteractiveTable from './shared/useInteractiveTable';
import SortIndicator from './shared/SortIndicator';

function toneForEvidence(level) {
  if (level === 'established') return 'var(--accent-green)';
  if (level === 'building') return 'var(--accent-blue)';
  if (level === 'sparse') return 'var(--accent-yellow)';
  return 'var(--accent-red)';
}

function metricText(value, digits = 3) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
  return Number(value).toFixed(digits);
}

function Badge({ label, tone }) {
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      padding: '2px 7px',
      borderRadius: 999,
      border: '1px solid var(--border)',
      background: 'var(--bg-tertiary)',
      color: tone || 'var(--text-secondary)',
      fontSize: 10,
      fontWeight: 700,
      textTransform: 'uppercase',
      letterSpacing: 0.3,
    }}>
      {label}
    </span>
  );
}

function TemplateRow({ row }) {
  const coverage = row.screening_metric_coverage || {};
  return (
    <div style={{ padding: '10px 0', borderBottom: '1px solid var(--border)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'flex-start', marginBottom: 6 }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)' }}>{row.name}</div>
            <Badge label={row.evidence_level || 'unknown'} tone={toneForEvidence(row.evidence_level)} />
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>
            {row.n_used} runs · S0 {fmtPct(row.s0_rate, 0)} · S0.5 {fmtPct(row.s05_rate, 0)} · S1 {fmtPct(row.s1_rate, 0)}
            {row.top_failure_reason ? ` · top fail ${row.top_failure_reason}` : ''}
          </div>
        </div>
        <div style={{ textAlign: 'right', fontSize: 10, color: 'var(--text-muted)' }}>
          <div>Train {fmtLoss(row.avg_loss_ratio)}</div>
          <div>Val {fmtLoss(row.avg_validation_loss_ratio)}</div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, minmax(0, 1fr))', gap: 8, marginBottom: 7, fontSize: 10 }}>
        <div><span style={{ color: 'var(--text-muted)' }}>Ind</span> <span style={{ color: 'var(--text-primary)' }}>{metricText(row.avg_induction_auc)}</span></div>
        <div><span style={{ color: 'var(--text-muted)' }}>Bind</span> <span style={{ color: 'var(--text-primary)' }}>{metricText(row.avg_binding_auc)}</span></div>
        <div><span style={{ color: 'var(--text-muted)' }}>AR</span> <span style={{ color: 'var(--text-primary)' }}>{metricText(row.avg_ar_auc)}</span></div>
        <div><span style={{ color: 'var(--text-muted)' }}>Hella</span> <span style={{ color: 'var(--text-primary)' }}>{metricText(row.avg_hellaswag_acc)}</span></div>
        <div><span style={{ color: 'var(--text-muted)' }}>Slots</span> <span style={{ color: 'var(--text-primary)' }}>{fmtNumber(row.slot_count)}</span></div>
      </div>

      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 6 }}>
        Coverage: ind {fmtNumber(coverage.induction)} · bind {fmtNumber(coverage.binding)} · ar {fmtNumber(coverage.associative_recall)} · hella {fmtNumber(coverage.hellaswag)} · wiki {fmtNumber(coverage.wikitext)}
      </div>

      {Array.isArray(row.diagnosis) && row.diagnosis.length > 0 && (
        <div style={{ fontSize: 11, color: 'var(--text-primary)', lineHeight: 1.5, marginBottom: 4 }}>
          Why: {row.diagnosis.join(' ')}
        </div>
      )}
      {Array.isArray(row.actions) && row.actions.length > 0 && (
        <div style={{ fontSize: 11, color: 'var(--accent-blue)', lineHeight: 1.5 }}>
          Change: {row.actions.join(' ')}
        </div>
      )}
    </div>
  );
}

const TEMPLATE_COLUMNS = [
  { key: 'name', label: 'Template' },
  { key: 'evidence_level', label: 'Evidence' },
  { key: 'n_used', label: 'Runs' },
  { key: 's0_rate', label: 'S0' },
  { key: 's05_rate', label: 'S0.5' },
  { key: 's1_rate', label: 'S1' },
  { key: 'avg_loss_ratio', label: 'Train LR' },
  { key: 'avg_validation_loss_ratio', label: 'Val LR' },
  { key: 'avg_induction_auc', label: 'Ind' },
  { key: 'avg_binding_auc', label: 'Bind' },
  { key: 'avg_hellaswag_acc', label: 'Hella' },
  { key: 'top_failure_reason', label: 'Issue' },
];

const SLOT_COLUMNS = [
  { key: 'slot_key', label: 'Slot' },
  { key: 'template_name', label: 'Template' },
  { key: 'slot_index', label: '#' },
  { key: 'n_used', label: 'Uses' },
  { key: 's1_rate', label: 'S1' },
  { key: 'avg_loss_ratio', label: 'Train LR' },
  { key: 'top_selected_motif', label: 'Selected' },
  { key: 'top_failure_reason', label: 'Issue' },
];

const EVIDENCE_ORDER = { insufficient: 0, sparse: 1, building: 2, established: 3 };

function getTemplateSortValue(row, key) {
  if (key === 'evidence_level') return EVIDENCE_ORDER[row.evidence_level] ?? -1;
  return row[key];
}

function getTemplateInitialSortDesc(key) {
  return key !== 'name' && key !== 'top_failure_reason';
}

function TemplateTable({ rows }) {
  const [search, setSearch] = useState('');
  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return rows || [];
    return (rows || []).filter((row) => {
      const text = [
        row.name,
        row.evidence_level,
        row.top_failure_reason,
        ...(row.diagnosis || []),
        ...(row.actions || []),
      ].filter(Boolean).join(' ').toLowerCase();
      return text.includes(q);
    });
  }, [rows, search]);

  const { sortKey, sortDesc, sortedRows, handleSort } = useInteractiveTable({
    rows: filtered,
    filterFields: [],
    initialSortKey: 'n_used',
    initialSortDesc: true,
    getSortValue: getTemplateSortValue,
    getInitialSortDesc: getTemplateInitialSortDesc,
  });

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, marginBottom: 10 }}>
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search templates..."
          style={{
            flex: 1,
            maxWidth: 280,
            background: 'var(--bg-tertiary)',
            border: '1px solid var(--border)',
            borderRadius: 6,
            color: 'var(--text-primary)',
            padding: '7px 10px',
            fontSize: 12,
          }}
        />
        <div style={{ fontSize: 11, color: 'var(--text-muted)', alignSelf: 'center' }}>
          {fmtNumber(sortedRows.length)} rows
        </div>
      </div>
      <div style={{ overflowX: 'auto', maxHeight: 560, overflowY: 'auto' }}>
        <table className="data-table" style={{ fontSize: 12 }}>
          <thead style={{ position: 'sticky', top: 0, zIndex: 1, background: 'var(--bg-primary)' }}>
            <tr>
              {TEMPLATE_COLUMNS.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  style={{ cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap', background: 'var(--bg-primary)' }}
                >
                  {col.label}
                  <SortIndicator active={sortKey === col.key} desc={sortDesc} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedRows.map((row) => (
              <tr key={row.name}>
                <td style={{ fontFamily: 'monospace', fontWeight: 600 }}>{row.name}</td>
                <td><Badge label={row.evidence_level || 'unknown'} tone={toneForEvidence(row.evidence_level)} /></td>
                <td style={{ textAlign: 'right' }}>{fmtNumber(row.n_used)}</td>
                <td style={{ textAlign: 'right' }}>{fmtPct(row.s0_rate, 0)}</td>
                <td style={{ textAlign: 'right' }}>{fmtPct(row.s05_rate, 0)}</td>
                <td style={{ textAlign: 'right', color: (row.s1_rate || 0) < 0.15 ? 'var(--accent-red)' : 'var(--text-primary)' }}>{fmtPct(row.s1_rate, 1)}</td>
                <td style={{ textAlign: 'right' }}>{fmtLoss(row.avg_loss_ratio)}</td>
                <td style={{ textAlign: 'right' }}>{fmtLoss(row.avg_validation_loss_ratio)}</td>
                <td style={{ textAlign: 'right' }}>{metricText(row.avg_induction_auc)}</td>
                <td style={{ textAlign: 'right' }}>{metricText(row.avg_binding_auc)}</td>
                <td style={{ textAlign: 'right' }}>{metricText(row.avg_hellaswag_acc)}</td>
                <td style={{ fontSize: 11, color: 'var(--text-muted)', maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {row.top_failure_reason || ''}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SlotTable({ rows }) {
  const [search, setSearch] = useState('');
  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return rows || [];
    return (rows || []).filter((row) => {
      const text = [
        row.slot_key,
        row.template_name,
        row.top_selected_motif,
        row.top_failure_reason,
        ...(row.slot_classes || []),
      ].filter(Boolean).join(' ').toLowerCase();
      return text.includes(q);
    });
  }, [rows, search]);

  const { sortKey, sortDesc, sortedRows, handleSort } = useInteractiveTable({
    rows: filtered,
    filterFields: [],
    initialSortKey: 'template_name',
    initialSortDesc: false,
  });

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, marginBottom: 10 }}>
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search slots or templates..."
          style={{
            flex: 1,
            maxWidth: 320,
            background: 'var(--bg-tertiary)',
            border: '1px solid var(--border)',
            borderRadius: 6,
            color: 'var(--text-primary)',
            padding: '7px 10px',
            fontSize: 12,
          }}
        />
        <div style={{ fontSize: 11, color: 'var(--text-muted)', alignSelf: 'center' }}>
          {fmtNumber(sortedRows.length)} rows
        </div>
      </div>
      <div style={{ overflowX: 'auto', maxHeight: 420, overflowY: 'auto' }}>
        <table className="data-table" style={{ fontSize: 12 }}>
          <thead style={{ position: 'sticky', top: 0, zIndex: 1, background: 'var(--bg-primary)' }}>
            <tr>
              {SLOT_COLUMNS.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  style={{ cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap', background: 'var(--bg-primary)' }}
                >
                  {col.label}
                  <SortIndicator active={sortKey === col.key} desc={sortDesc} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedRows.map((row) => (
              <tr key={row.slot_key}>
                <td style={{ fontFamily: 'monospace', fontWeight: 600 }}>{row.slot_key}</td>
                <td style={{ fontFamily: 'monospace' }}>{row.template_name}</td>
                <td style={{ textAlign: 'right' }}>{fmtNumber(row.slot_index)}</td>
                <td style={{ textAlign: 'right' }}>{fmtNumber(row.n_used)}</td>
                <td style={{ textAlign: 'right' }}>{fmtPct(row.s1_rate, 1)}</td>
                <td style={{ textAlign: 'right' }}>{fmtLoss(row.avg_loss_ratio)}</td>
                <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>{row.top_selected_motif || ''}</td>
                <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>{row.top_failure_reason || ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function TemplateSlotObservability() {
  const { summary } = useAriaData() || {};
  const data = summary?.template_observability;
  if (!data) return null;

  const topTemplates = Array.isArray(data.top_templates) ? data.top_templates.slice(0, 4) : [];
  const strugglingTemplates = Array.isArray(data.struggling_templates) ? data.struggling_templates.slice(0, 4) : [];
  const slots = Array.isArray(data.slot_observability) ? data.slot_observability.slice(0, 5) : [];
  const motifs = Array.isArray(data.motif_slots) ? data.motif_slots.slice(0, 4) : [];
  const recommendations = Array.isArray(data.recommendations) ? data.recommendations : [];
  const allTemplates = Array.isArray(data.all_templates) ? data.all_templates : [];
  const allSlots = Array.isArray(data.all_slots) ? data.all_slots : [];
  const loss = data.loss_distribution || {};
  const overview = data.summary || {};

  return (
    <div className="card">
      <div className="card-title">Template & Slot Observability</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 14, lineHeight: 1.5 }}>
        Tracks template families, slot pressure points, and screening-task evidence so you can distinguish sparse data from genuine weakness.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 10, marginBottom: 16 }}>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(overview.templates_tracked || 0)}</div>
          <div className="stat-label">Active Templates</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{fmtNumber(overview.templates_observed_total || 0)} observed total</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(overview.insufficient_templates || 0)}</div>
          <div className="stat-label">Insufficient</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Need more runs before ranking</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(overview.sparse_templates || 0)}</div>
          <div className="stat-label">Sparse</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Partial evidence only</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(overview.established_templates || 0)}</div>
          <div className="stat-label">Established</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Enough samples to compare</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtLoss(loss.training?.median)}</div>
          <div className="stat-label">Median Train LR</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Val {fmtLoss(loss.validation?.median)}</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtNumber(overview.routing_fast_lane_positive_templates || 0)}</div>
          <div className="stat-label">Slow Starters</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{fmtNumber(overview.routing_fast_lane_templates || 0)} fast-lane templates</div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
        <div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase' }}>
            Highest Success Templates
          </div>
          {topTemplates.length > 0 ? topTemplates.map((row) => (
            <TemplateRow key={row.name} row={row} />
          )) : <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>No template data yet.</div>}
        </div>

        <div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase' }}>
            Templates To Fix
          </div>
          {strugglingTemplates.length > 0 ? strugglingTemplates.map((row) => (
            <TemplateRow key={row.name} row={row} />
          )) : <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>No struggling templates identified.</div>}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 0.9fr', gap: 18, marginTop: 18 }}>
        <div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase' }}>
            Weakest Slots
          </div>
          {slots.length > 0 ? slots.map((row) => (
            <div key={row.slot_key} style={{ padding: '8px 0', borderBottom: '1px solid var(--border)', fontSize: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, marginBottom: 4 }}>
                <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{row.slot_key}</span>
                <span style={{ color: 'var(--accent-red)' }}>{fmtPct(row.s1_rate)}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, color: 'var(--text-muted)', fontSize: 11 }}>
                <span>{row.n_used} uses · motif {row.top_selected_motif || 'none'}</span>
                <span>LR {fmtLoss(row.avg_loss_ratio)}</span>
              </div>
              <div style={{ marginTop: 2, color: 'var(--text-muted)', fontSize: 10 }}>
                {row.template_name} · classes {(row.slot_classes || []).join(', ') || 'unknown'}
                {row.top_failure_reason ? ` · fail ${row.top_failure_reason}` : ''}
              </div>
            </div>
          )) : <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>No explicit slot telemetry yet.</div>}
          {motifs.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 6, fontWeight: 600, textTransform: 'uppercase' }}>
                Supporting Motif Aggregates
              </div>
              {motifs.map((row) => (
                <div key={row.name} style={{ display: 'flex', justifyContent: 'space-between', gap: 8, padding: '4px 0', borderBottom: '1px solid var(--border)', fontSize: 11 }}>
                  <span style={{ color: 'var(--text-primary)' }}>{row.name}</span>
                  <span style={{ color: 'var(--text-muted)' }}>{row.n_used} uses</span>
                  <span style={{ color: 'var(--accent-blue)' }}>{fmtPct(row.s1_rate)}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase' }}>
            What Needs Improvement
          </div>
          {recommendations.length > 0 ? recommendations.map((item, idx) => (
            <div key={idx} style={{ padding: '9px 10px', marginBottom: 8, background: 'var(--bg-tertiary)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 12, color: 'var(--text-primary)', lineHeight: 1.5 }}>
              {item}
            </div>
          )) : <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>No recommendations yet.</div>}
        </div>
      </div>

      <div style={{ marginTop: 18 }}>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase' }}>
            Active Templates
        </div>
        {allTemplates.length > 0 ? <TemplateTable rows={allTemplates} /> : <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>No templates observed yet.</div>}
      </div>

      <div style={{ marginTop: 18 }}>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase' }}>
            All Slots
        </div>
        {allSlots.length > 0 ? <SlotTable rows={allSlots} /> : <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>No explicit slot telemetry yet.</div>}
      </div>
    </div>
  );
}
