import React from 'react';
import { lossColor, noveltyColor } from '../../utils/colors';
import MetricRow from '../program/MetricRow';
import BenchmarkEvidenceSnapshot from '../program/BenchmarkEvidenceSnapshot';
import RobustnessProfile from '../program/RobustnessProfile';
import AriaAdvice from '../program/AriaAdvice';
import ReferenceComparison from '../program/ReferenceComparison';
import ExternalBenchmarkCard from '../program/ExternalBenchmarkCard';
import TrainingCurve from '../program/TrainingCurve';
import HypothesisInfo from '../program/HypothesisInfo';

/**
 * Left column of the two-column grid: core metrics, benchmarks, timing.
 */
export function CoreMetricsColumn({ program, leaderboardEntry, refineAnalysis, fmt, fmtMs, fmtMem, fmtInt }) {
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
      <AriaAdvice analysis={refineAnalysis} />
      <ReferenceComparison program={program} leaderboardEntry={leaderboardEntry} />
      <ExternalBenchmarkCard program={program} />
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
