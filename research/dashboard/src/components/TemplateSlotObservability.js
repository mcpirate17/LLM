import React from 'react';
import { useAriaData } from '../hooks/useAriaData';
import { fmtPct as _fmtPct, fmtLoss } from '../utils/format';

const fmtPct = (v) => _fmtPct(v, 1);

function TemplateRow({ row, tone = 'good' }) {
  const color = tone === 'bad' ? 'var(--accent-red)' : tone === 'warn' ? 'var(--accent-yellow)' : 'var(--accent-green)'
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 0.8fr 0.8fr 1fr', gap: 10, padding: '8px 0', borderBottom: '1px solid var(--border)', alignItems: 'center' }}>
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {row.name}
        </div>
        <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
          {row.n_used} runs · {row.slot_count || 0} inferred slots
          {row.top_failure_reason ? ` · top fail: ${row.top_failure_reason}` : ''}
          {row.routing_fast_lane_runs ? ` · fast lane ${fmtPct(row.routing_fast_lane_positive_rate)}` : ''}
        </div>
      </div>
      <div style={{ fontSize: 12, color, fontWeight: 700 }}>{fmtPct(row.s1_rate)}</div>
      <div style={{ fontSize: 12, color: 'var(--text-primary)' }}>{fmtLoss(row.avg_validation_loss_ratio ?? row.avg_loss_ratio)}</div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{fmtLoss(row.best_loss_ratio)}</div>
    </div>
  )
}

export default function TemplateSlotObservability() {
  const { summary } = useAriaData() || {}
  const data = summary?.template_observability
  if (!data) return null

  const topTemplates = Array.isArray(data.top_templates) ? data.top_templates.slice(0, 4) : []
  const strugglingTemplates = Array.isArray(data.struggling_templates) ? data.struggling_templates.slice(0, 4) : []
  const slots = Array.isArray(data.slot_observability) ? data.slot_observability.slice(0, 5) : []
  const motifs = Array.isArray(data.motif_slots) ? data.motif_slots.slice(0, 4) : []
  const recommendations = Array.isArray(data.recommendations) ? data.recommendations : []
  const loss = data.loss_distribution || {}
  const overview = data.summary || {}

  return (
    <div className="card">
      <div className="card-title">Template & Slot Observability</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 14, lineHeight: 1.5 }}>
        Tracks which structural templates and motif slots are producing survivors, how their loss curves look, and where the search space is wasting budget.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 10, marginBottom: 16 }}>
        <div className="stat-card">
          <div className="stat-value">{Number(overview.templates_tracked || 0)}</div>
          <div className="stat-label">Templates Tracked</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{Number(overview.avg_templates_per_graph || 0).toFixed(2)} templates/graph</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{Number(overview.motifs_tracked || 0)}</div>
          <div className="stat-label">Motifs Tracked</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{Number(overview.avg_motifs_per_graph || 0).toFixed(2)} motifs/graph</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtLoss(loss.training?.median)}</div>
          <div className="stat-label">Median Train LR</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>P75 {fmtLoss(loss.training?.p75)}</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{fmtLoss(loss.validation?.median)}</div>
          <div className="stat-label">Median Val LR</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>P75 {fmtLoss(loss.validation?.p75)}</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{Number(overview.routing_fast_lane_templates || 0)}</div>
          <div className="stat-label">Routing Fast-Lane Templates</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            {Number(overview.routing_fast_lane_positive_templates || 0)} with positive probes
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
        <div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase' }}>
            Highest Success Templates
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 0.8fr 0.8fr 1fr', gap: 10, fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
            <span>Template</span>
            <span>S1</span>
            <span>Avg LR</span>
            <span>Best</span>
          </div>
          {topTemplates.length > 0 ? topTemplates.map((row) => (
            <TemplateRow key={row.name} row={row} tone="good" />
          )) : <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>No template data yet.</div>}
        </div>

        <div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase' }}>
            Templates To Fix
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 0.8fr 0.8fr 1fr', gap: 10, fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
            <span>Template</span>
            <span>S1</span>
            <span>Avg LR</span>
            <span>Best</span>
          </div>
          {strugglingTemplates.length > 0 ? strugglingTemplates.map((row) => (
            <TemplateRow key={row.name} row={row} tone="bad" />
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
          {Array.isArray(overview.zero_slot_templates) && overview.zero_slot_templates.length > 0 && (
            <div style={{ marginTop: 10, fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
              Zero-slot templates detected: {overview.zero_slot_templates.slice(0, 4).join(', ')}
              {overview.zero_slot_templates.length > 4 ? ` +${overview.zero_slot_templates.length - 4}` : ''}
            </div>
          )}
          {Array.isArray(overview.inactive_template_names) && overview.inactive_template_names.length > 0 && (
            <div style={{ marginTop: 10, fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
              Inactive templates hidden from rankings: {overview.inactive_template_names.slice(0, 4).join(', ')}
              {overview.inactive_template_names.length > 4 ? ` +${overview.inactive_template_names.length - 4}` : ''}
            </div>
          )}
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
    </div>
  )
}
