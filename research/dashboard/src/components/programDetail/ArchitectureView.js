import React from 'react';
import FingerprintRadar from '../program/FingerprintRadar';
import MetricRow from '../program/MetricRow';
import TokenMixingTaxonomy from '../program/TokenMixingTaxonomy';
import GatingDiagnostics from '../program/GatingDiagnostics';
import SparsityDiagnostics from '../program/SparsityDiagnostics';
import RefinementRationale from '../program/RefinementRationale';
import RefinementLineage from '../program/RefinementLineage';
import RefinementAdvisor from '../program/RefinementAdvisor';

/**
 * Right column of the two-column grid: fingerprint radar, CKA similarity, timing.
 */
export function FingerprintColumn({ program, fmtMs, fmtMem, fmtInt }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: -8 }}>
        Fingerprint & Similarity
      </div>
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 16,
        padding: '16px 12px',
        background: 'var(--bg-secondary)',
        borderRadius: 8,
        border: '1px solid var(--border)'
      }}>
        <FingerprintRadar program={program} size={260} />
        <div style={{ width: '100%', borderTop: '1px solid var(--border)', paddingTop: 12 }}>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
            Primary Reference Class
          </div>
          <div style={{ fontSize: 13, color: 'var(--accent-purple)', fontWeight: 600 }}>
            {program.most_similar_to || 'Truly Novel (No match)'}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
            Highest CKA similarity among the known baseline families.
          </div>
        </div>
        {(program.fp_cka_vs_transformer != null || program.fp_cka_vs_ssm != null) && (
          <div style={{ width: '100%', borderTop: '1px solid var(--border)', paddingTop: 12 }}>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8 }}>
              CKA Similarity vs Baselines
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
              Higher percentages mean this program behaves more like that baseline family.
            </div>
            {[
              { label: 'Transformer', value: program.fp_cka_vs_transformer, color: 'var(--accent-blue)' },
              { label: 'SSM', value: program.fp_cka_vs_ssm, color: 'var(--accent-green)' },
              { label: 'Conv', value: program.fp_cka_vs_conv, color: 'var(--accent-yellow)' },
            ].map(({ label, value, color }) => value != null && (
              <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                <span style={{ fontSize: 11, color: 'var(--text-secondary)', minWidth: 70 }}>{label}</span>
                <div style={{ flex: 1, height: 8, background: 'var(--bg-tertiary)', borderRadius: 4 }}>
                  <div style={{
                    width: `${Math.min(value, 1) * 100}%`, height: '100%',
                    background: color, borderRadius: 4, opacity: 0.7,
                  }} />
                </div>
                <span style={{ fontSize: 10, color: 'var(--text-muted)', minWidth: 30, textAlign: 'right' }}>{(Number(value) * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        )}
      </div>
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
 * Architecture analysis sections rendered below the two-column grid:
 * refinement, taxonomy, gating, sparsity, designer button, best training.
 */
function ArchitectureView({
  program, leaderboardEntry, resultId,
  refineAnalysis, refineAnalysisLoading, refineAnalysisError,
  handleLaunchRefinement, actionStarting,
  onViewInLeaderboard, onOpenInDesigner, onClose,
}) {
  return (
    <>
      <RefinementRationale program={program} />
      <RefinementLineage program={program} onViewInLeaderboard={onViewInLeaderboard} />
      {program.stage1_passed && (
        <RefinementAdvisor
          analysis={refineAnalysis}
          loading={refineAnalysisLoading}
          error={refineAnalysisError}
          onLaunchRefinement={handleLaunchRefinement}
          actionStarting={actionStarting}
        />
      )}
      <TokenMixingTaxonomy graphJson={program.graph_json_parsed} />
      <GatingDiagnostics program={program} />
      <SparsityDiagnostics program={program} />
      {program.graph_json_parsed && onOpenInDesigner && (
        <div style={{ marginTop: 8 }}>
          <button
            onClick={() => {
              onClose();
              onOpenInDesigner(resultId);
            }}
            style={{
              background: 'rgba(188, 140, 255, 0.15)',
              border: '1px solid rgba(188, 140, 255, 0.4)',
              color: 'var(--accent-purple)',
              fontSize: 12,
              fontWeight: 600,
              padding: '8px 20px',
              borderRadius: 6,
              cursor: 'pointer',
              width: '100%',
            }}
            title="Open this architecture in the visual graph designer"
          >
            Open in Designer
          </button>
        </div>
      )}
      {leaderboardEntry?.investigation_best_training && (
        <div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
            Best Training Program (from Investigation)
          </div>
          <pre style={{
            fontSize: 11, padding: 8, background: 'var(--bg-tertiary)',
            borderRadius: 4, overflow: 'auto', maxHeight: 120,
            color: 'var(--text-secondary)',
          }}>
            {typeof leaderboardEntry.investigation_best_training === 'string'
              ? leaderboardEntry.investigation_best_training
              : JSON.stringify(leaderboardEntry.investigation_best_training, null, 2)}
          </pre>
        </div>
      )}
    </>
  );
}

export default React.memo(ArchitectureView);
