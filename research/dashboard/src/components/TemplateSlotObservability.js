import React, { useEffect, useMemo, useState } from 'react';
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

function toneForCategory(cat) {
  if (cat === 'strong') return 'var(--accent-green)';
  if (cat === 'decent') return 'var(--accent-blue)';
  if (cat === 'data-sparse') return 'var(--accent-yellow)';
  if (cat === 'untested') return 'var(--text-muted)';
  if (cat === 'reference') return 'var(--accent-cyan, var(--accent-blue))';
  if (cat === 'exotic') return 'var(--accent-purple, var(--accent-yellow))';
  if (cat === 'weak') return 'var(--accent-red)';
  return 'var(--text-muted)';
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
      letterSpacing: 0,
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
            {row.structural_category && <Badge label={row.structural_category} tone={toneForCategory(row.structural_category)} />}
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

const CATEGORY_ORDER = {
  strong: 6,
  decent: 5,
  reference: 4,
  exotic: 3,
  'data-sparse': 2,
  untested: 1,
  weak: 0,
};

const TEMPLATE_COLUMNS = [
  { key: 'name', label: 'Template', tooltip: 'Structural recipe name.', sticky: true, left: 0, width: 250 },
  { key: 'structural_category', label: 'Label', tooltip: 'Template quality category from aggregate evidence.' },
  { key: 'evidence_level', label: 'Evidence', tooltip: 'Sample support level for comparing this template.' },
  { key: 'n_used', label: 'Runs', tooltip: 'Number of observed graphs using this template.' },
  { key: 's0_rate', label: 'S0', tooltip: 'Share of template runs that passed Stage 0.' },
  { key: 's05_rate', label: 'S0.5', tooltip: 'Share of template runs that passed stability screening.' },
  { key: 's1_rate', label: 'S1', tooltip: 'Share of Stage 0 passes that reached Stage 1.' },
  { key: 'avg_loss_ratio', label: 'Train LR', tooltip: 'Average training loss ratio.' },
  { key: 'avg_validation_loss_ratio', label: 'Val LR', tooltip: 'Average validation loss ratio.' },
  { key: 'avg_induction_auc', label: 'Ind', tooltip: 'Average induction-task AUC.' },
  { key: 'avg_binding_auc', label: 'Bind', tooltip: 'Average binding/copy-task AUC.' },
  { key: 'avg_hellaswag_acc', label: 'Hella', tooltip: 'Average HellaSwag accuracy signal.' },
  { key: 'top_failure_reason', label: 'Issue', tooltip: 'Most common failure reason.' },
];

const SLOT_COLUMNS = [
  { key: 'slot_key', label: 'Slot', tooltip: 'Template slot identifier.', sticky: true, left: 0, width: 270 },
  { key: 'template_name', label: 'Template', tooltip: 'Template that owns this slot.' },
  { key: 'slot_index', label: '#', tooltip: 'Slot index inside the template.' },
  { key: 'n_used', label: 'Uses', tooltip: 'Number of observed fills for this slot.' },
  { key: 's1_rate', label: 'S1', tooltip: 'Share of slot uses that reached Stage 1.' },
  { key: 'avg_loss_ratio', label: 'Train LR', tooltip: 'Average training loss ratio for this slot.' },
  { key: 'top_selected_motif', label: 'Selected', tooltip: 'Most frequently selected motif.' },
  { key: 'top_failure_reason', label: 'Issue', tooltip: 'Most common failure reason.' },
];

const TEMPLATE_GROUPS = [
  { label: 'Identity', span: 3 },
  { label: 'Health', span: 4 },
  { label: 'Learning', span: 2 },
  { label: 'Benchmarks', span: 3 },
  { label: 'Diagnosis', span: 1 },
];

const SLOT_GROUPS = [
  { label: 'Identity', span: 3 },
  { label: 'Health', span: 3 },
  { label: 'Diagnosis', span: 2 },
];

const TEMPLATE_FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'strong', label: 'Strong' },
  { key: 'decent', label: 'Decent' },
  { key: 'weak', label: 'Weak' },
  { key: 'sparse', label: 'Sparse' },
  { key: 'untested', label: 'Untested' },
  { key: 'established', label: 'Established' },
  { key: 'low_s1', label: 'Low S1' },
];

const TEMPLATE_SORT_PRESETS = [
  { key: 'most_runs', label: 'Most Runs', sortKey: 'n_used', desc: true },
  { key: 'worst_s1', label: 'Worst S1', sortKey: 's1_rate', desc: false },
  { key: 'best_val', label: 'Best Val', sortKey: 'avg_validation_loss_ratio', desc: false },
  { key: 'worst_gap', label: 'Worst Val Gap', sortKey: 'val_gap', desc: true },
  { key: 'best_ind', label: 'Best Ind', sortKey: 'avg_induction_auc', desc: true },
  { key: 'best_hella', label: 'Best Hella', sortKey: 'avg_hellaswag_acc', desc: true },
];

const SLOT_FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'low_s1', label: 'Low S1' },
  { key: 'high_loss', label: 'High Loss' },
  { key: 'role', label: 'Role Slots' },
  { key: 'has_issue', label: 'Has Issue' },
];

const SLOT_SORT_PRESETS = [
  { key: 'template', label: 'Template Order', sortKey: 'template_name', desc: false },
  { key: 'most_used', label: 'Most Used', sortKey: 'n_used', desc: true },
  { key: 'worst_s1', label: 'Worst S1', sortKey: 's1_rate', desc: false },
  { key: 'worst_loss', label: 'Worst Loss', sortKey: 'avg_loss_ratio', desc: true },
];

const EVIDENCE_ORDER = { insufficient: 0, sparse: 1, building: 2, established: 3 };

function getTemplateSortValue(row, key) {
  if (key === 'evidence_level') return EVIDENCE_ORDER[row.evidence_level] ?? -1;
  if (key === 'structural_category') return CATEGORY_ORDER[row.structural_category] ?? -1;
  if (key === 'val_gap') {
    const val = Number(row.avg_validation_loss_ratio);
    const train = Number(row.avg_loss_ratio);
    if (!Number.isFinite(val) || !Number.isFinite(train)) return null;
    return val - train;
  }
  return row[key];
}

function getTemplateInitialSortDesc(key) {
  return key !== 'name' && key !== 'top_failure_reason';
}

function stickyCellStyle(col, extra = {}) {
  if (!col?.sticky) return extra;
  return {
    ...extra,
    position: 'sticky',
    left: col.left,
    minWidth: col.width,
    maxWidth: col.width,
    width: col.width,
    background: extra.background || 'var(--bg-primary)',
    zIndex: extra.zIndex || 2,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    boxShadow: '1px 0 0 var(--border)',
  };
}

function valLossTone(row) {
  const val = Number(row.avg_validation_loss_ratio);
  const train = Number(row.avg_loss_ratio);
  if (!Number.isFinite(val)) return 'var(--text-muted)';
  if (val >= 0.68 || (Number.isFinite(train) && val > train * 1.15)) return 'var(--accent-yellow)';
  return 'var(--text-primary)';
}

function templateMatchesFilter(row, filter) {
  if (filter === 'all') return true;
  if (filter === 'established') return row.evidence_level === 'established';
  if (filter === 'sparse') return row.evidence_level === 'sparse' || row.structural_category === 'data-sparse';
  if (filter === 'low_s1') return row.s1_rate != null && row.s1_rate < 0.15;
  return row.structural_category === filter;
}

function slotMatchesFilter(row, filter) {
  if (filter === 'all') return true;
  if (filter === 'low_s1') return row.s1_rate != null && row.s1_rate < 0.15;
  if (filter === 'high_loss') return row.avg_loss_ratio != null && row.avg_loss_ratio > 0.65;
  if (filter === 'role') return (row.slot_classes || []).some((item) => typeof item === 'string' && item.startsWith('role:'));
  if (filter === 'has_issue') return Boolean(row.top_failure_reason);
  return true;
}

function controlStyle(maxWidth) {
  return {
    flex: '1 1 180px',
    maxWidth,
    background: 'var(--bg-tertiary)',
    border: '1px solid var(--border)',
    borderRadius: 6,
    color: 'var(--text-primary)',
    padding: '7px 10px',
    fontSize: 12,
  };
}

function filterButtonStyle(active) {
  return {
    padding: '3px 10px',
    fontSize: 11,
    borderRadius: 12,
    cursor: 'pointer',
    background: active ? 'var(--accent-blue)22' : 'var(--bg-tertiary)',
    color: active ? 'var(--accent-blue)' : 'var(--text-muted)',
    border: `1px solid ${active ? 'var(--accent-blue)' : 'var(--border)'}`,
    fontWeight: active ? 600 : 400,
  };
}

function TemplateTable({ rows }) {
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('all');
  const [sortPreset, setSortPreset] = useState('most_runs');
  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return (rows || []).filter((row) => {
      if (!templateMatchesFilter(row, filter)) return false;
      if (!q) return true;
      const text = [
        row.name,
        row.structural_category,
        row.evidence_level,
        row.top_failure_reason,
        ...(row.diagnosis || []),
        ...(row.actions || []),
      ].filter(Boolean).join(' ').toLowerCase();
      return text.includes(q);
    });
  }, [filter, rows, search]);

  const { sortKey, sortDesc, setSortKey, setSortDesc, sortedRows, handleSort } = useInteractiveTable({
    rows: filtered,
    filterFields: [],
    initialSortKey: 'n_used',
    initialSortDesc: true,
    getSortValue: getTemplateSortValue,
    getInitialSortDesc: getTemplateInitialSortDesc,
  });

  useEffect(() => {
    const preset = TEMPLATE_SORT_PRESETS.find((item) => item.key === sortPreset);
    if (!preset) return;
    setSortKey(preset.sortKey);
    setSortDesc(preset.desc);
  }, [setSortDesc, setSortKey, sortPreset]);

  const hasFilters = filter !== 'all' || sortPreset !== 'most_runs' || search.trim() !== '';

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, marginBottom: 10, flexWrap: 'wrap' }}>
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search templates..."
          style={controlStyle(280)}
        />
        <select value={sortPreset} onChange={(e) => setSortPreset(e.target.value)} style={{ ...controlStyle(180), cursor: 'pointer', flex: '0 0 180px' }}>
          {TEMPLATE_SORT_PRESETS.map((preset) => (
            <option key={preset.key} value={preset.key}>{preset.label}</option>
          ))}
        </select>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', alignSelf: 'center' }}>
          {fmtNumber(sortedRows.length)} of {fmtNumber((rows || []).length)} rows
        </div>
      </div>
      <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap', marginBottom: 10 }}>
        {TEMPLATE_FILTERS.map((item) => (
          <button key={item.key} onClick={() => setFilter(item.key)} style={filterButtonStyle(filter === item.key)}>
            {item.label}
          </button>
        ))}
        <button
          disabled={!hasFilters}
          onClick={() => {
            setSearch('');
            setFilter('all');
            setSortPreset('most_runs');
          }}
          style={{
            ...filterButtonStyle(false),
            background: 'var(--bg-secondary)',
            color: hasFilters ? 'var(--text-primary)' : 'var(--text-muted)',
            cursor: hasFilters ? 'pointer' : 'not-allowed',
          }}
        >
          Reset
        </button>
      </div>
      <div style={{ overflowX: 'auto', maxHeight: 560, overflowY: 'auto' }}>
        <table className="data-table" style={{ fontSize: 12, borderCollapse: 'separate', borderSpacing: 0, minWidth: 1280 }}>
          <thead style={{ position: 'sticky', top: 0, zIndex: 1, background: 'var(--bg-primary)' }}>
            <tr>
              {TEMPLATE_GROUPS.map((group) => (
                <th
                  key={group.label}
                  colSpan={group.span}
                  style={{
                    textAlign: 'center',
                    fontSize: 10,
                    color: 'var(--text-muted)',
                    textTransform: 'uppercase',
                    letterSpacing: 0,
                    background: 'var(--bg-primary)',
                    position: 'sticky',
                    top: 0,
                    zIndex: 3,
                  }}
                >
                  {group.label}
                </th>
              ))}
            </tr>
            <tr>
              {TEMPLATE_COLUMNS.map((col) => (
                <th
                  key={col.key}
                  title={col.tooltip}
                  onClick={() => handleSort(col.key)}
                  style={stickyCellStyle(col, {
                    cursor: 'pointer',
                    userSelect: 'none',
                    whiteSpace: 'nowrap',
                    background: 'var(--bg-primary)',
                    top: 34,
                    zIndex: col.sticky ? 4 : 1,
                  })}
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
                <td style={stickyCellStyle(TEMPLATE_COLUMNS[0], { fontFamily: 'monospace', fontWeight: 600, background: 'var(--bg-secondary)', zIndex: 3, whiteSpace: 'nowrap' })}>{row.name}</td>
                <td><Badge label={row.structural_category || '?'} tone={toneForCategory(row.structural_category)} /></td>
                <td><Badge label={row.evidence_level || 'unknown'} tone={toneForEvidence(row.evidence_level)} /></td>
                <td style={{ textAlign: 'right' }}>{fmtNumber(row.n_used)}</td>
                <td style={{ textAlign: 'right' }}>{fmtPct(row.s0_rate, 0)}</td>
                <td style={{ textAlign: 'right' }}>{fmtPct(row.s05_rate, 0)}</td>
                <td style={{ textAlign: 'right', color: (row.s1_rate || 0) < 0.15 ? 'var(--accent-red)' : 'var(--text-primary)' }}>{fmtPct(row.s1_rate, 1)}</td>
                <td style={{ textAlign: 'right' }}>{fmtLoss(row.avg_loss_ratio)}</td>
                <td
                  title={Number.isFinite(Number(row.avg_validation_loss_ratio)) && Number.isFinite(Number(row.avg_loss_ratio)) ? `Gap ${(Number(row.avg_validation_loss_ratio) - Number(row.avg_loss_ratio)).toFixed(3)}` : undefined}
                  style={{ textAlign: 'right', color: valLossTone(row) }}
                >
                  {fmtLoss(row.avg_validation_loss_ratio)}
                </td>
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
  const [filter, setFilter] = useState('all');
  const [sortPreset, setSortPreset] = useState('template');
  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return (rows || []).filter((row) => {
      if (!slotMatchesFilter(row, filter)) return false;
      if (!q) return true;
      const text = [
        row.slot_key,
        row.template_name,
        row.top_selected_motif,
        row.top_failure_reason,
        ...(row.slot_classes || []),
      ].filter(Boolean).join(' ').toLowerCase();
      return text.includes(q);
    });
  }, [filter, rows, search]);

  const { sortKey, sortDesc, setSortKey, setSortDesc, sortedRows, handleSort } = useInteractiveTable({
    rows: filtered,
    filterFields: [],
    initialSortKey: 'template_name',
    initialSortDesc: false,
  });

  useEffect(() => {
    const preset = SLOT_SORT_PRESETS.find((item) => item.key === sortPreset);
    if (!preset) return;
    setSortKey(preset.sortKey);
    setSortDesc(preset.desc);
  }, [setSortDesc, setSortKey, sortPreset]);

  const hasFilters = filter !== 'all' || sortPreset !== 'template' || search.trim() !== '';

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, marginBottom: 10, flexWrap: 'wrap' }}>
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search slots or templates..."
          style={controlStyle(320)}
        />
        <select value={sortPreset} onChange={(e) => setSortPreset(e.target.value)} style={{ ...controlStyle(180), cursor: 'pointer', flex: '0 0 180px' }}>
          {SLOT_SORT_PRESETS.map((preset) => (
            <option key={preset.key} value={preset.key}>{preset.label}</option>
          ))}
        </select>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', alignSelf: 'center' }}>
          {fmtNumber(sortedRows.length)} of {fmtNumber((rows || []).length)} rows
        </div>
      </div>
      <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap', marginBottom: 10 }}>
        {SLOT_FILTERS.map((item) => (
          <button key={item.key} onClick={() => setFilter(item.key)} style={filterButtonStyle(filter === item.key)}>
            {item.label}
          </button>
        ))}
        <button
          disabled={!hasFilters}
          onClick={() => {
            setSearch('');
            setFilter('all');
            setSortPreset('template');
          }}
          style={{
            ...filterButtonStyle(false),
            background: 'var(--bg-secondary)',
            color: hasFilters ? 'var(--text-primary)' : 'var(--text-muted)',
            cursor: hasFilters ? 'pointer' : 'not-allowed',
          }}
        >
          Reset
        </button>
      </div>
      <div style={{ overflowX: 'auto', maxHeight: 420, overflowY: 'auto' }}>
        <table className="data-table" style={{ fontSize: 12, borderCollapse: 'separate', borderSpacing: 0, minWidth: 960 }}>
          <thead style={{ position: 'sticky', top: 0, zIndex: 1, background: 'var(--bg-primary)' }}>
            <tr>
              {SLOT_GROUPS.map((group) => (
                <th
                  key={group.label}
                  colSpan={group.span}
                  style={{
                    textAlign: 'center',
                    fontSize: 10,
                    color: 'var(--text-muted)',
                    textTransform: 'uppercase',
                    letterSpacing: 0,
                    background: 'var(--bg-primary)',
                    position: 'sticky',
                    top: 0,
                    zIndex: 3,
                  }}
                >
                  {group.label}
                </th>
              ))}
            </tr>
            <tr>
              {SLOT_COLUMNS.map((col) => (
                <th
                  key={col.key}
                  title={col.tooltip}
                  onClick={() => handleSort(col.key)}
                  style={stickyCellStyle(col, {
                    cursor: 'pointer',
                    userSelect: 'none',
                    whiteSpace: 'nowrap',
                    background: 'var(--bg-primary)',
                    top: 34,
                    zIndex: col.sticky ? 4 : 1,
                  })}
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
                <td style={stickyCellStyle(SLOT_COLUMNS[0], { fontFamily: 'monospace', fontWeight: 600, background: 'var(--bg-secondary)', zIndex: 3, whiteSpace: 'nowrap' })}>{row.slot_key}</td>
                <td style={{ fontFamily: 'monospace' }}>{row.template_name}</td>
                <td style={{ textAlign: 'right' }}>{fmtNumber(row.slot_index)}</td>
                <td style={{ textAlign: 'right' }}>{fmtNumber(row.n_used)}</td>
                <td style={{ textAlign: 'right', color: (row.s1_rate || 0) < 0.15 ? 'var(--accent-red)' : 'var(--text-primary)' }}>{fmtPct(row.s1_rate, 1)}</td>
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

      <RoleSlotRollup rows={allSlots} />

      <div style={{ marginTop: 18 }}>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase' }}>
            All Slots
        </div>
        {allSlots.length > 0 ? <SlotTable rows={allSlots} /> : <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>No explicit slot telemetry yet.</div>}
      </div>
    </div>
  );
}

/**
 * Aggregates slot telemetry by capability role (``role:trunk_compression``,
 * ``role:global_retrieval``, etc.). These role slots are emitted by the
 * capability-first templates added 2026-04-16 and ride on the existing
 * ``slot_classes`` channel. Summing across slot_keys for the same role
 * gives a clearer "which slot is the bottleneck" signal than reading the
 * All Slots table one-by-one.
 */
function RoleSlotRollup({ rows }) {
  const roleAgg = useMemo(() => {
    if (!rows || rows.length === 0) return [];
    const byRole = new Map();
    for (const row of rows) {
      const classes = row.slot_classes || [];
      const role = classes.find((c) => typeof c === 'string' && c.startsWith('role:'));
      if (!role) continue;
      const key = role.slice('role:'.length);
      const bucket = byRole.get(key) || {
        role: key,
        n_used: 0,
        n_s1: 0,
        loss_sum: 0,
        loss_n: 0,
        motifs: {},
        templates: new Set(),
      };
      const n = row.n_used || 0;
      bucket.n_used += n;
      bucket.n_s1 += (row.n_stage1 || Math.round((row.s1_rate || 0) * n)) || 0;
      if (typeof row.avg_loss_ratio === 'number' && isFinite(row.avg_loss_ratio) && n > 0) {
        bucket.loss_sum += row.avg_loss_ratio * n;
        bucket.loss_n += n;
      }
      const motif = row.top_selected_motif;
      if (motif) {
        bucket.motifs[motif] = (bucket.motifs[motif] || 0) + n;
      }
      if (row.template_name) bucket.templates.add(row.template_name);
      byRole.set(key, bucket);
    }
    return Array.from(byRole.values()).map((b) => {
      const s1_rate = b.n_used > 0 ? b.n_s1 / b.n_used : 0;
      const avg_loss = b.loss_n > 0 ? b.loss_sum / b.loss_n : null;
      const topMotif = Object.entries(b.motifs).sort((a, z) => z[1] - a[1])[0];
      return {
        role: b.role,
        n_used: b.n_used,
        s1_rate,
        avg_loss,
        top_motif: topMotif ? `${topMotif[0]} (${topMotif[1]})` : 'n/a',
        n_templates: b.templates.size,
      };
    }).sort((a, b) => b.n_used - a.n_used);
  }, [rows]);

  if (roleAgg.length === 0) return null;

  return (
    <div style={{ marginTop: 18 }}>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase' }}>
        Role Slots (capability-first)
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, lineHeight: 1.5 }}>
        Aggregates <code>role:*</code> slot telemetry emitted by role-slot templates (<code>typed_slot_memory_block</code>, <code>*_retrieval_v2</code>, etc.). Each row sums across every template instance of the role. Use this to spot whether a specific capability slot (e.g. <code>global_retrieval</code>) is dragging the search down.
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ color: 'var(--text-secondary)', textTransform: 'uppercase', fontSize: 10 }}>
              <th style={{ textAlign: 'left', padding: '4px 8px' }}>Role</th>
              <th style={{ textAlign: 'right', padding: '4px 8px' }}>Uses</th>
              <th style={{ textAlign: 'right', padding: '4px 8px' }}>S1 Rate</th>
              <th style={{ textAlign: 'right', padding: '4px 8px' }}>Avg Loss</th>
              <th style={{ textAlign: 'left', padding: '4px 8px' }}>Top Motif</th>
              <th style={{ textAlign: 'right', padding: '4px 8px' }}>Templates</th>
            </tr>
          </thead>
          <tbody>
            {roleAgg.map((row) => (
              <tr key={row.role} style={{ borderTop: '1px solid var(--border)' }}>
                <td style={{ padding: '4px 8px', fontFamily: 'monospace', fontWeight: 600, color: 'var(--accent-green, var(--text-primary))' }}>{row.role}</td>
                <td style={{ padding: '4px 8px', textAlign: 'right' }}>{row.n_used}</td>
                <td style={{ padding: '4px 8px', textAlign: 'right', color: row.s1_rate < 0.15 ? 'var(--accent-red)' : 'var(--text-primary)' }}>{fmtPct(row.s1_rate, 1)}</td>
                <td style={{ padding: '4px 8px', textAlign: 'right' }}>{fmtLoss(row.avg_loss)}</td>
                <td style={{ padding: '4px 8px', fontFamily: 'monospace', fontSize: 11, color: 'var(--text-muted)' }}>{row.top_motif}</td>
                <td style={{ padding: '4px 8px', textAlign: 'right', color: 'var(--text-muted)' }}>{row.n_templates}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
