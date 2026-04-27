import React from 'react';
import { lossColor, noveltyColor, pplColor, hellaswagColor, blimpColor, probeAucColor } from '../../utils/colors';
import { scoreColor } from '../../utils/format';
import MetricRow from '../program/MetricRow';
import BenchmarkEvidenceSnapshot from '../program/BenchmarkEvidenceSnapshot';
import RobustnessProfile from '../program/RobustnessProfile';
import AriaAdvice from '../program/AriaAdvice';
import ReferenceComparison from '../program/ReferenceComparison';
import ExternalBenchmarkCard from '../program/ExternalBenchmarkCard';
import TrainingCurve from '../program/TrainingCurve';
import HypothesisInfo from '../program/HypothesisInfo';

const TONE_COLOR = {
  positive: 'var(--accent-green)',
  neutral: 'var(--accent-yellow)',
  negative: 'var(--accent-red)',
};

function finite(value) {
  if (value == null) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function firstMetric(...values) {
  for (const value of values) {
    const n = finite(value);
    if (n != null) return n;
  }
  return null;
}

function valueFor(program, leaderboardEntry, ...keys) {
  for (const key of keys) {
    const value = firstMetric(program?.[key], leaderboardEntry?.[key], program?.leaderboard?.[key]);
    if (value != null) return value;
  }
  return null;
}

function toneHigher(value, good, neutral) {
  const n = finite(value);
  if (n == null) return null;
  if (n >= good) return 'positive';
  if (n >= neutral) return 'neutral';
  return 'negative';
}

function toneLower(value, good, neutral) {
  const n = finite(value);
  if (n == null) return null;
  if (n <= good) return 'positive';
  if (n <= neutral) return 'neutral';
  return 'negative';
}

function MetricValue({ value, digits = 3, suffix = '', tone, note, title }) {
  if (value == null) return null;
  const formatted = typeof value === 'number' ? value.toFixed(digits) : String(value);
  const color = TONE_COLOR[tone] || 'var(--text-primary)';
  return (
    <span title={title} style={{ color, fontWeight: tone === 'positive' || tone === 'negative' ? 650 : 500 }}>
      {formatted}{suffix}
      {tone && (
        <span style={{
          marginLeft: 6,
          fontSize: 10,
          color,
          textTransform: 'uppercase',
          letterSpacing: 0,
        }}>
          {tone}
        </span>
      )}
      {note && <span style={{ marginLeft: 6, fontSize: 11, color: 'var(--text-muted)' }}>{note}</span>}
    </span>
  );
}

export function BenchmarkTestMatrix({ program, leaderboardEntry }) {
  const externalBenchmarks = program?.external_benchmarks || program?.external_benchmarks_json_parsed || {};
  const longCtx = externalBenchmarks?.long_context || {};
  const tests = [
    {
      group: 'Language',
      rows: [
        {
          key: 'wikitext_perplexity',
          label: 'WikiText PPL',
          value: valueFor(program, leaderboardEntry, 'wikitext_perplexity', 'wikitext_ppl'),
          digits: 1,
          tone: value => toneLower(value, 25, 120),
          color: pplColor,
          note: 'lower is better',
          title: 'Real-token WikiText perplexity after tiktoken/BPE rescreen. Lower is better.',
        },
        {
          key: 'wikitext_score',
          label: 'WikiText score',
          value: valueFor(program, leaderboardEntry, 'wikitext_score'),
          digits: 3,
          tone: value => toneHigher(value, 0.55, 0.25),
          note: 'higher is better',
        },
        {
          key: 'tinystories_perplexity',
          label: 'TinyStories PPL',
          value: valueFor(program, leaderboardEntry, 'tinystories_perplexity'),
          digits: 1,
          tone: value => toneLower(value, 25, 120),
          note: 'lower is better',
        },
        {
          key: 'tinystories_score',
          label: 'TinyStories score',
          value: valueFor(program, leaderboardEntry, 'tinystories_score'),
          digits: 3,
          tone: value => toneHigher(value, 0.55, 0.25),
          note: 'higher is better',
        },
        {
          key: 'hellaswag_acc',
          label: 'HellaSwag',
          value: valueFor(program, leaderboardEntry, 'hellaswag_acc'),
          digits: 3,
          tone: value => toneHigher(value, 0.25, 0.18),
          color: hellaswagColor,
          note: 'higher is better',
          title: 'Commonsense multiple-choice accuracy. Higher is better.',
        },
        {
          key: 'blimp_overall_accuracy',
          label: 'BLiMP',
          value: valueFor(program, leaderboardEntry, 'blimp_overall_accuracy'),
          digits: 3,
          tone: value => toneHigher(value, 0.50, 0.45),
          color: blimpColor,
          note: 'higher is better',
          title: 'BLiMP grammatical acceptability accuracy. Higher is better.',
        },
      ],
    },
    {
      group: 'Capability Probes',
      rows: [
        {
          key: 'induction_v2_investigation_auc',
          label: 'Induction v2',
          value: valueFor(program, leaderboardEntry, 'induction_v2_investigation_auc'),
          digits: 3,
          tone: value => toneHigher(value, 0.45, 0.20),
          color: probeAucColor,
          note: 'higher is better',
        },
        {
          key: 'induction_auc',
          label: 'Induction v1',
          value: valueFor(program, leaderboardEntry, 'induction_auc'),
          digits: 3,
          tone: value => toneHigher(value, 0.45, 0.20),
          color: probeAucColor,
          note: 'higher is better',
        },
        {
          key: 'binding_v2_investigation_auc',
          label: 'Binding v2',
          value: valueFor(program, leaderboardEntry, 'binding_v2_investigation_auc'),
          digits: 3,
          tone: value => toneHigher(value, 0.45, 0.20),
          color: probeAucColor,
          note: 'higher is better',
        },
        {
          key: 'binding_auc',
          label: 'Binding v1',
          value: valueFor(program, leaderboardEntry, 'binding_auc'),
          digits: 3,
          tone: value => toneHigher(value, 0.45, 0.20),
          color: probeAucColor,
          note: 'higher is better',
        },
        {
          key: 'ar_auc',
          label: 'Associative recall',
          value: valueFor(program, leaderboardEntry, 'ar_auc'),
          digits: 3,
          tone: value => toneHigher(value, 0.45, 0.10),
          color: probeAucColor,
          note: 'higher is better',
        },
      ],
    },
    {
      group: 'Generalization',
      rows: [
        {
          key: 'loss_ratio',
          label: 'Loss ratio',
          value: valueFor(program, leaderboardEntry, 'loss_ratio', 'screening_loss_ratio'),
          digits: 4,
          tone: value => toneLower(value, 0.70, 1.00),
          note: 'lower is better',
        },
        {
          key: 'baseline_loss_ratio',
          label: 'Baseline ratio',
          value: valueFor(program, leaderboardEntry, 'baseline_loss_ratio', 'validation_baseline_ratio'),
          digits: 4,
          tone: value => toneLower(value, 1.00, 1.10),
          note: '< 1 beats baseline',
        },
        {
          key: 'validation_loss_ratio',
          label: 'Validation LR',
          value: valueFor(program, leaderboardEntry, 'validation_loss_ratio'),
          digits: 4,
          tone: value => toneLower(value, 0.70, 1.00),
          note: 'lower is better',
        },
        {
          key: 'generalization_gap',
          label: 'Generalization gap',
          value: valueFor(program, leaderboardEntry, 'generalization_gap'),
          digits: 4,
          tone: value => toneLower(value, 0.10, 0.35),
          note: 'lower is better',
        },
      ],
    },
    {
      group: 'Robustness',
      rows: [
        {
          key: 'investigation_robustness',
          label: 'Investigation robustness',
          value: valueFor(program, leaderboardEntry, 'investigation_robustness'),
          digits: 3,
          tone: value => toneHigher(value, 0.80, 0.50),
          note: 'higher is better',
        },
        {
          key: 'robustness_noise_score',
          label: 'Noise sensitivity',
          value: valueFor(program, leaderboardEntry, 'robustness_noise_score'),
          digits: 3,
          tone: value => toneLower(value, 0.15, 0.35),
          note: 'lower is better',
        },
        {
          key: 'robustness_long_ctx_score',
          label: 'Long-context score',
          value: firstMetric(valueFor(program, leaderboardEntry, 'robustness_long_ctx_score'), longCtx.long_context_score),
          digits: 3,
          tone: value => toneHigher(value, 0.70, 0.35),
          note: 'higher is better',
        },
        {
          key: 'robustness_long_ctx_scaling_score',
          label: 'LC scaling',
          value: firstMetric(valueFor(program, leaderboardEntry, 'robustness_long_ctx_scaling_score'), longCtx.scaling_score, longCtx.long_context_score),
          digits: 3,
          tone: value => toneHigher(value, 0.70, 0.35),
          note: 'higher is better',
        },
        {
          key: 'robustness_long_ctx_assoc_score',
          label: 'LC assoc retrieval',
          value: firstMetric(valueFor(program, leaderboardEntry, 'robustness_long_ctx_assoc_score'), longCtx.assoc_retrieval_score),
          digits: 3,
          tone: value => toneHigher(value, 0.70, 0.35),
          note: 'higher is better',
        },
        {
          key: 'robustness_long_ctx_multi_hop_score',
          label: 'LC multi-hop',
          value: firstMetric(valueFor(program, leaderboardEntry, 'robustness_long_ctx_multi_hop_score'), longCtx.multi_hop_score),
          digits: 3,
          tone: value => toneHigher(value, 0.70, 0.35),
          note: 'higher is better',
        },
        {
          key: 'robustness_long_ctx_passkey_score',
          label: 'LC passkey',
          value: firstMetric(valueFor(program, leaderboardEntry, 'robustness_long_ctx_passkey_score'), longCtx.passkey_score),
          digits: 3,
          tone: value => toneHigher(value, 0.70, 0.35),
          note: 'higher is better',
        },
        {
          key: 'robustness_long_ctx_combined_score',
          label: 'LC combined',
          value: firstMetric(valueFor(program, leaderboardEntry, 'robustness_long_ctx_combined_score'), longCtx.combined_score, longCtx.long_context_score),
          digits: 3,
          tone: value => toneHigher(value, 0.70, 0.35),
          note: 'higher is better',
        },
      ],
    },
    {
      group: 'Fingerprint Trajectory',
      rows: [
        {
          key: 'fp_jacobian_erf_density',
          label: 'ERF density',
          value: valueFor(program, leaderboardEntry, 'fp_jacobian_erf_density'),
          digits: 3,
          tone: value => toneHigher(value, 0.50, 0.20),
          note: 'higher is better',
        },
        {
          key: 'fp_id_collapse_rate',
          label: 'ID collapse rate',
          value: valueFor(program, leaderboardEntry, 'fp_id_collapse_rate'),
          digits: 4,
          tone: value => toneLower(value, -0.01, 0.01),
          note: 'lower is better',
        },
        {
          key: 'fp_jacobian_erf_decay_slope',
          label: 'ERF decay slope',
          value: valueFor(program, leaderboardEntry, 'fp_jacobian_erf_decay_slope'),
          digits: 4,
          tone: value => toneLower(value, -0.05, 0.02),
          note: 'more negative is better',
        },
        {
          key: 'fp_logit_margin_velocity',
          label: 'Logit margin velocity',
          value: valueFor(program, leaderboardEntry, 'fp_logit_margin_velocity'),
          digits: 4,
          tone: value => toneHigher(value, 0.005, 0.001),
          note: 'higher is better',
        },
        {
          key: 'fp_icld_velocity',
          label: 'ICLD velocity',
          value: valueFor(program, leaderboardEntry, 'fp_icld_velocity'),
          digits: 4,
          tone: value => toneLower(value, -0.02, 0.00),
          note: 'more negative is better',
        },
        {
          key: 'fp_jacobian_erf_variance',
          label: 'ERF variance',
          value: valueFor(program, leaderboardEntry, 'fp_jacobian_erf_variance'),
          digits: 1,
          tone: value => toneHigher(value, 50000, 5000),
          note: 'higher is better',
        },
      ],
    },
    {
      group: 'Efficiency',
      rows: [
        {
          key: 'composite_score',
          label: 'Composite score',
          value: valueFor(program, leaderboardEntry, 'composite_score'),
          digits: 1,
          tone: value => toneHigher(value, 205, 150),
          color: scoreColor,
          note: 'higher is better',
        },
        {
          key: 'param_efficiency',
          label: 'Parameter efficiency',
          value: firstMetric(
            valueFor(program, leaderboardEntry, 'param_efficiency'),
            externalBenchmarks.best_param_efficiency,
            externalBenchmarks.scaling_comparison?.best_param_efficiency,
          ),
          digits: 4,
          tone: value => toneHigher(value, 0.50, 0.10),
          note: 'higher is better',
        },
        {
          key: 'sample_efficiency',
          label: 'Sample efficiency',
          value: valueFor(program, leaderboardEntry, 'sample_efficiency'),
          digits: 3,
          tone: value => toneHigher(value, 0.80, 0.50),
          note: 'higher is better',
        },
        {
          key: 'activation_sparsity_score',
          label: 'Activation sparsity',
          value: valueFor(program, leaderboardEntry, 'activation_sparsity_score'),
          digits: 3,
          tone: value => toneHigher(value, 0.70, 0.35),
          note: 'higher is better',
        },
        {
          key: 'compression_ratio',
          label: 'Compression ratio',
          value: valueFor(program, leaderboardEntry, 'compression_ratio'),
          digits: 3,
          tone: value => toneHigher(value, 0.70, 0.25),
          note: 'higher is better',
        },
        {
          key: 'quant_int8_retention',
          label: 'INT8 retention',
          value: valueFor(program, leaderboardEntry, 'quant_int8_retention'),
          digits: 3,
          tone: value => toneHigher(value, 0.90, 0.75),
          note: 'higher is better',
        },
      ],
    },
  ];

  const populated = tests
    .map(group => ({
      ...group,
      rows: group.rows.filter(row => row.value != null),
    }))
    .filter(group => group.rows.length > 0);

  if (populated.length === 0) return null;

  const allRows = populated.flatMap(group => group.rows);
  const counts = allRows.reduce((acc, row) => {
    const tone = row.tone?.(row.value) || 'neutral';
    acc[tone] = (acc[tone] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="card" style={{ padding: 12, marginTop: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'baseline', marginBottom: 8, flexWrap: 'wrap' }}>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
          Evaluation Tests
        </div>
        <div style={{ display: 'flex', gap: 8, fontSize: 10, color: 'var(--text-muted)', flexWrap: 'wrap' }}>
          {['positive', 'neutral', 'negative'].map(tone => (
            <span key={tone} style={{ color: TONE_COLOR[tone] }}>
              {counts[tone] || 0} {tone}
            </span>
          ))}
        </div>
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10 }}>
        Every recorded test with direction-of-goodness applied.
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 14 }}>
        {populated.map(group => (
          <div key={group.group} style={{ minWidth: 0 }}>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4, fontWeight: 600 }}>
              {group.group}
            </div>
            {group.rows.map(row => {
              const tone = row.tone?.(row.value) || 'neutral';
              return (
                <MetricRow
                  key={row.key}
                  label={row.label}
                  title={row.title || `${row.note || ''}`}
                  value={(
                    <MetricValue
                      value={row.value}
                      digits={row.digits}
                      tone={tone}
                      note={row.note}
                      title={row.title}
                    />
                  )}
                />
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Left column of the two-column grid: core metrics, benchmarks, timing.
 */
export function CoreMetricsColumn({ program, leaderboardEntry, fmt, fmtMs, fmtMem, fmtInt }) {
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
        Core Metrics
      </div>
      <div style={{ fontSize: 13 }}>
        <MetricRow label="Parameters" value={program.param_count ? `${(program.param_count / 1e6).toFixed(2)}M` : null} />
        <MetricRow label="Loss Ratio" value={program.loss_ratio != null ?
          <span style={{
            color: lossColor(program.loss_ratio),
            fontWeight: program.loss_ratio < 0.5 ? 600 : 'normal',
          }} title={program.loss_ratio < 0.5 ? 'Learned quickly — strong candidate' : program.loss_ratio < 0.7 ? 'Moderate learning' : 'Slow learning'}>
            {fmt(program.loss_ratio)}
          </span> : null} />
        <MetricRow label="Final Loss" value={fmt(program.final_loss)} />
        <MetricRow label="Discovery Loss" value={program.discovery_loss != null ? fmt(program.discovery_loss) : null} />
        <MetricRow label="Discovery LR" value={program.discovery_loss_ratio != null ? fmt(program.discovery_loss_ratio) : null} />
        <MetricRow label="Validation Loss" value={program.validation_loss != null ? fmt(program.validation_loss) : null} />
        <MetricRow label="Validation LR" value={program.validation_loss_ratio != null ? fmt(program.validation_loss_ratio) : null} />
        <MetricRow label="Gen Gap" value={program.generalization_gap != null ? fmt(program.generalization_gap) : null} />
        <MetricRow label="Baseline Ratio" value={program.baseline_loss_ratio != null ?
          <span style={{
            color: program.baseline_loss_ratio < 1 ? 'var(--accent-green)' : 'var(--accent-red)',
            fontWeight: program.baseline_loss_ratio < 1 ? 600 : 'normal',
          }} title={program.baseline_loss_ratio < 1 ? 'Beats a standard transformer!' : 'Underperforms a transformer of same size'}>
            {fmt(program.baseline_loss_ratio)} {program.baseline_loss_ratio < 1 ? '(beats transformer)' : ''}
          </span> : null} />
        <MetricRow label="HellaSwag" value={program.hellaswag_acc != null ? fmt(program.hellaswag_acc, 3) : null} />
        <MetricRow label="Induction AUC" value={program.induction_auc != null ? fmt(program.induction_auc, 3) : null} />
        <MetricRow label="Induction v2" value={program.induction_v2_investigation_auc != null ?
          <span title="Investigation-tier mixed-gap probe (500 steps, median-of-3 seeds). Overrides v1 in the binding composite when present.">
            {fmt(program.induction_v2_investigation_auc, 3)}
            {program.induction_auc != null && (
              <span style={{ marginLeft: 6, fontSize: 11, color: 'var(--text-muted)' }}>
                (Δ{(program.induction_v2_investigation_auc - program.induction_auc).toFixed(3)} vs v1)
              </span>
            )}
          </span> : null} />
        <MetricRow label="AR AUC" value={program.ar_auc != null ? fmt(program.ar_auc, 3) : null} />
        <MetricRow label="Binding AUC" value={program.binding_auc != null ? fmt(program.binding_auc, 3) : null} />
        <MetricRow label="Binding v2" value={program.binding_v2_investigation_auc != null ?
          <span title="Investigation-tier extended-budget probe (2400 steps, 5 distances incl. 64, median-of-3 seeds). Overrides v1 in the binding composite when present.">
            {fmt(program.binding_v2_investigation_auc, 3)}
            {program.binding_auc != null && (
              <span style={{ marginLeft: 6, fontSize: 11, color: 'var(--text-muted)' }}>
                (Δ{(program.binding_v2_investigation_auc - program.binding_auc).toFixed(3)} vs v1)
              </span>
            )}
          </span> : null} />
        <MetricRow label="BLiMP" value={program.blimp_overall_accuracy != null ? fmt(program.blimp_overall_accuracy, 3) : null} />
        <MetricRow label="WikiText PPL" value={program.wikitext_perplexity != null ? fmt(program.wikitext_perplexity, 1) : null} />
        <MetricRow label="Composite" value={(program.composite_score ?? leaderboardEntry?.composite_score) != null ?
          fmt(program.composite_score ?? leaderboardEntry?.composite_score, 1) : null} />
        <MetricRow label="Throughput" value={program.throughput_tok_s != null ? `${Number(program.throughput_tok_s).toFixed(0)} tok/s` : null} />
        <MetricRow label="Param Efficiency" value={program.param_efficiency != null ? fmt(program.param_efficiency) : (leaderboardEntry?.param_efficiency != null ? fmt(leaderboardEntry.param_efficiency) : null)} />
        <MetricRow label="Sample Efficiency" value={program.sample_efficiency != null ?
          <span style={{
            color: program.sample_efficiency >= 0.8 ? 'var(--accent-green)' : program.sample_efficiency >= 0.5 ? 'var(--accent-yellow)' : 'var(--accent-red)',
          }} title={`Converges to 25% initial loss in ${((1 - program.sample_efficiency) * 100).toFixed(0)}% of training budget`}>
            {fmt(program.sample_efficiency, 3)}
          </span> : null} />
        <MetricRow label="Novelty" value={program.novelty_score != null ?
          <span style={{
            color: noveltyColor(program.novelty_score),
          }} title={program.novelty_score > 0.8 ? 'Very different from known architectures' : program.novelty_score > 0.5 ? 'Moderately novel' : 'Similar to existing architectures'}>
            {fmt(program.novelty_score, 3)}
          </span> : null} />
      </div>
      <BenchmarkEvidenceSnapshot program={program} leaderboardEntry={leaderboardEntry} />
      <RobustnessProfile program={program} leaderboardEntry={leaderboardEntry} />
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8, marginTop: 12 }}>
        Sandbox Timing
      </div>
      <div style={{ fontSize: 13 }}>
        <MetricRow label="Compile" value={fmtMs(program.compile_time_ms)} />
        <MetricRow label="Forward" value={fmtMs(program.forward_time_ms)} />
        <MetricRow label="Backward" value={fmtMs(program.backward_time_ms)} />
        <MetricRow label="Peak Memory" value={fmtMem(program.peak_memory_mb)} />
        <MetricRow label="FLOPs (fwd)" value={program.flops_forward ? fmtInt(program.flops_forward) : null} />
      </div>
    </div>
  );
}

export function ProgramSupportCards({ program, leaderboardEntry, refineAnalysis }) {
  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
      gap: 16,
      alignItems: 'start',
    }}>
      <AriaAdvice analysis={refineAnalysis} />
      <ReferenceComparison program={program} leaderboardEntry={leaderboardEntry} />
      <ExternalBenchmarkCard program={program} />
    </div>
  );
}

/**
 * Training stats, training curve, LLM analysis, hypothesis/decision info.
 * Rendered below the two-column grid.
 */
function TrainingMetricsPanel({ program, resultId, linkedHypothesis, linkedDecision, fmt, fmtMs }) {
  return (
    <>
      {program.initial_loss != null && (
        <div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
            Training Metrics
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 13 }}>
            <MetricRow label="Initial Loss" value={fmt(program.initial_loss)} />
            <MetricRow label="Min Loss" value={fmt(program.min_loss)} />
            <MetricRow label="Steps" value={program.n_train_steps} />
            <MetricRow label="Avg Step Time" value={fmtMs(program.avg_step_time_ms)} />
            <MetricRow label="Mean Grad Norm" value={fmt(program.mean_grad_norm, 3)} />
            <MetricRow label="Max Grad Norm" value={fmt(program.max_grad_norm, 3)} />
          </div>
        </div>
      )}
      {program.has_training_curve && (
        <TrainingCurve resultId={resultId} />
      )}
      {program.llm_explanation && (
        <div style={{
          padding: 12,
          background: 'var(--bg-tertiary)',
          borderRadius: 4,
          borderLeft: '2px solid var(--accent-purple)',
          fontSize: 13,
          color: 'var(--text-secondary)',
          fontStyle: 'italic',
        }}>
          <div style={{ fontSize: 11, color: 'var(--accent-purple)', marginBottom: 4, fontWeight: 600, fontStyle: 'normal' }}>
            ARIA'S ANALYSIS
          </div>
          {program.llm_explanation}
        </div>
      )}
      {linkedHypothesis && (
        <HypothesisInfo hypothesis={linkedHypothesis} />
      )}
      {linkedDecision && (
        <div style={{
          padding: 12, background: 'var(--bg-tertiary)', borderRadius: 4,
          borderLeft: `2px solid ${
            linkedDecision.decision_type === 'go' ? 'var(--accent-green)' :
            linkedDecision.decision_type === 'no_go' ? 'var(--accent-red)' : 'var(--accent-yellow)'
          }`,
          fontSize: 13,
        }}>
          <div style={{
            fontSize: 11, fontWeight: 600, textTransform: 'uppercase', marginBottom: 4,
            color: linkedDecision.decision_type === 'go' ? 'var(--accent-green)' :
                   linkedDecision.decision_type === 'no_go' ? 'var(--accent-red)' : 'var(--accent-yellow)',
          }}>
            Decision: {linkedDecision.decision_type?.replace('_', ' ')}
          </div>
          <div>{linkedDecision.rationale}</div>
        </div>
      )}
    </>
  );
}

export default React.memo(TrainingMetricsPanel);
