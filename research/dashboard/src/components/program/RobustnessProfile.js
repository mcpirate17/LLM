import React from 'react';
import MetricRow from './MetricRow';

export function RobustnessProfile({ program, leaderboardEntry }) {
  const source = leaderboardEntry || program || {};
  const noise = source?.robustness_noise_score;
  const longCtx = source?.robustness_long_ctx_score;
  const externalBenchmarks = program?.external_benchmarks || program?.external_benchmarks_json_parsed || {};
  const longCtxBench = externalBenchmarks?.long_context || {};
  const longCtxScalingScore = longCtxBench?.scaling_score;
  const longCtxAssocScore = longCtxBench?.assoc_retrieval_score;
  const longCtxMultiHopScore = longCtxBench?.multi_hop_score;
  const longCtxPasskeyScore = longCtxBench?.passkey_score;
  const longCtxRetrievalAgg = longCtxBench?.retrieval_aggregate_score;
  const longCtxCombined = longCtxBench?.combined_score ?? longCtxBench?.long_context_score;
  const longCtxMaxViable = longCtxBench?.scaling?.max_viable_len;
  const longCtxBenchmarkVersion = longCtxBench?.benchmark_version;
  const initStd = source?.init_sensitivity_std;
  const quantRetentionRaw = source?.quant_int8_retention;
  const quantRetention = quantRetentionRaw == null
    ? null
    : (Number(quantRetentionRaw) <= 1 ? Number(quantRetentionRaw) * 100 : Number(quantRetentionRaw));
  const qualityPerByte = source?.quant_quality_per_byte;
  const spectralCandidates = [
    program?.jacobian_spectral_norm,
    program?.fp_jacobian_spectral_norm,
    source?.jacobian_spectral_norm,
    source?.fp_jacobian_spectral_norm,
    source?.fp_spectral_norm,
    source?.spectral_norm,
  ];
  const spectralNorm = (() => {
    for (const candidate of spectralCandidates) {
      if (candidate == null) continue;
      const num = Number(candidate);
      if (Number.isFinite(num) && num > 0) return num;
    }
    return null;
  })();

  const hasAny = [
    noise,
    longCtx,
    longCtxScalingScore,
    longCtxAssocScore,
    longCtxMultiHopScore,
    longCtxPasskeyScore,
    longCtxRetrievalAgg,
    longCtxCombined,
    longCtxMaxViable,
    initStd,
    quantRetention,
    qualityPerByte,
    spectralNorm,
  ]
    .some(v => v != null && Number.isFinite(Number(v)));
  if (!hasAny) return null;

  const gauge = (value, invert = false) => {
    if (value == null) return null;
    const v = Math.max(0, Math.min(1, Number(value)));
    const pct = (invert ? (1 - v) : v) * 100;
    return `${pct.toFixed(0)}%`;
  };

  return (
    <div style={{
      marginTop: 12,
      padding: 10,
      background: 'var(--bg-tertiary)',
      borderRadius: 6,
      border: '1px solid var(--border)',
    }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600, marginBottom: 8 }}>
        Robustness Profile
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 12 }}>
        <MetricRow label="Noise sensitivity" value={noise != null ? `${Number(noise).toFixed(3)} (${gauge(noise, true)})` : null} />
        <MetricRow label="Long-context score" value={longCtx != null ? `${Number(longCtx).toFixed(3)} (${gauge(longCtx, false)})` : null} />
        <MetricRow label="LongCtx scaling probe" value={longCtxScalingScore != null ? Number(longCtxScalingScore).toFixed(3) : null} />
        <MetricRow label="LongCtx assoc retrieval" value={longCtxAssocScore != null ? Number(longCtxAssocScore).toFixed(3) : null} />
        <MetricRow label="LongCtx multi-hop" value={longCtxMultiHopScore != null ? Number(longCtxMultiHopScore).toFixed(3) : null} />
        <MetricRow label="LongCtx passkey" value={longCtxPasskeyScore != null ? Number(longCtxPasskeyScore).toFixed(3) : null} />
        <MetricRow label="LongCtx retrieval agg" value={longCtxRetrievalAgg != null ? Number(longCtxRetrievalAgg).toFixed(3) : null} />
        <MetricRow label="LongCtx combined" value={longCtxCombined != null ? Number(longCtxCombined).toFixed(3) : null} />
        <MetricRow label="LongCtx max viable len" value={longCtxMaxViable != null ? String(longCtxMaxViable) : null} />
        <MetricRow label="LongCtx benchmark" value={longCtxBenchmarkVersion || null} />
        <MetricRow label="Init sensitivity std" value={initStd != null ? Number(initStd).toFixed(4) : null} />
        <MetricRow label="INT8 retention" value={quantRetention != null ? `${quantRetention.toFixed(1)}%` : null} />
        <MetricRow label="Quality per byte" value={qualityPerByte != null ? Number(qualityPerByte).toFixed(4) : null} />
        <MetricRow label="Spectral norm" value={spectralNorm != null ? Number(spectralNorm).toFixed(4) : null} />
      </div>
    </div>
  );
}

export default React.memo(RobustnessProfile);
