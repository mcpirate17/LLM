import React, { useState, useEffect, useMemo } from 'react';
import useCopyToClipboard from '../hooks/useCopyToClipboard';
import { discoveryScore, discoveryScoreBreakdown } from '../utils/scores';
import { scoreColor } from '../utils/format';
import { reliabilityColor } from '../utils/colors';

const API_BASE = process.env.REACT_APP_API_URL || '';

const COMPRESSION_FACTORS = {
  low_rank: 0.55,
  shared_basis: 0.5,
  hash_trick: 0.35,
  structured_sparse: 0.4,
  kronecker: 0.5,
  polynomial: 0.6,
  residual_quantized: 0.3,
  compressed_attention: 0.7,
};

function parseArchSpec(value) {
  if (!value || typeof value !== 'string') return null;
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch {
    return null;
  }
}

function compressionSummary(program) {
  const spec = parseArchSpec(program.arch_spec_json);
  const compressionKey = spec?.choices?.weight_storage || spec?.choices?.token_representation;
  const factor = COMPRESSION_FACTORS[compressionKey] || 1.0;
  const rawParams = program.param_count || program.graph_n_params_estimate || null;
  const compressedParams = rawParams != null ? Math.max(1, Math.round(rawParams * factor)) : null;
  const ratio = rawParams != null && compressedParams != null
    ? Math.max(0.01, Math.min(1.0, compressedParams / rawParams))
    : null;
  const memoryMb = compressedParams != null ? (compressedParams * 4) / (1024 * 1024) : null;
  const qualityRetention = program.baseline_loss_ratio != null
    ? Math.max(0, Math.min(1, 1.25 - program.baseline_loss_ratio))
    : program.loss_ratio != null
      ? Math.max(0, Math.min(1, 1.0 - program.loss_ratio))
      : null;
  return {
    label: compressionKey || 'dense',
    ratio,
    memoryMb,
    qualityRetention,
  };
}

function metricChips(program) {
  const chips = [];
  chips.push({ label: 'Loss', source: 'measured', reliability: program.loss_ratio != null ? 'high' : 'low' });
  chips.push({
    label: 'Novelty',
    source: program.cka_source === 'artifact' ? 'artifact-backed' : 'heuristic',
    reliability: program.novelty_confidence != null
      ? (program.novelty_confidence >= 0.7 ? 'high' : program.novelty_confidence >= 0.4 ? 'medium' : 'low')
      : 'low',
  });
  chips.push({
    label: 'Baseline',
    source: program.baseline_loss_ratio != null ? 'baseline-run' : 'not-available',
    reliability: program.baseline_loss_ratio != null ? 'medium' : 'low',
  });
  if (program.routing_confidence_mean != null) {
    chips.push({
      label: 'Routing',
      source: 'telemetry',
      reliability: program.routing_confidence_mean >= 0.7 ? 'high' : program.routing_confidence_mean >= 0.4 ? 'medium' : 'low',
    });
  }
  return chips;
}


const QKV_OPS = new Set(['local_window_attn', 'sliding_window_mask', 'multi_head_mix']);

const TOKEN_MIXING_FAMILIES = {
  local_window_attn: 'attention',
  sliding_window_mask: 'attention',
  softmax_last: 'attention',
  multi_head_mix: 'attention',
  selective_scan: 'ssm',
  conv1d_seq: 'conv',
  rfft_seq: 'frequency',
  irfft_seq: 'frequency',
  sort_seq: 'sorting',
  argsort_seq: 'sorting',
  token_pool_restore: 'pooling',
  cumsum_seq: 'pooling',
  roll_seq: 'pooling',
  basis_expansion: 'functional',
  integral_kernel: 'functional',
  fixed_point_iter: 'functional',
};

const FAMILY_LABELS = {
  attention: 'QKV-based',
  ssm: 'State Space',
  conv: 'Convolution',
  frequency: 'Frequency Domain',
  sorting: 'Sort-based',
  pooling: 'Pooling',
  functional: 'Functional/Operator',
};

const FAMILY_COLORS = {
  attention: 'var(--accent-blue)',
  ssm: 'var(--accent-green)',
  conv: 'var(--accent-yellow)',
  frequency: 'var(--accent-purple)',
  sorting: 'var(--accent-red)',
  pooling: 'var(--text-muted)',
  functional: '#e0a060',
};

/** Classify a program's token mixing families from graph_json. */
function classifyTokenMixing(program) {
  const raw = program.graph_json || program._graph_json;
  if (!raw) return { families: new Set(), qkvFree: null, ops: [] };
  try {
    const graph = typeof raw === 'string' ? JSON.parse(raw) : raw;
    const nodes = graph.nodes || {};
    const ops = Object.values(nodes).map(n => n.op_name || n.op).filter(Boolean);
    const families = new Set();
    const detectedOps = [];
    for (const op of ops) {
      const family = TOKEN_MIXING_FAMILIES[op];
      if (family) {
        families.add(family);
        detectedOps.push(op);
      }
    }
    const qkvFree = !ops.some(op => QKV_OPS.has(op));
    return { families, qkvFree, ops: detectedOps };
  } catch {
    return { families: new Set(), qkvFree: null, ops: [] };
  }
}

function qkvUsageDescriptor(program) {
  const usage = program?.qkv_usage;
  if (usage === 'qkv_free') {
    return {
      label: 'QKV-free',
      detail: 'No Q/K/V attention path; relies on alternatives like SSM/conv/frequency/functional.',
      color: 'var(--accent-green)',
    };
  }
  if (usage === 'q_eq_k_eq_v') {
    return {
      label: 'Q=K=V',
      detail: 'Shared-projection attention variant (reduced attention parameterization).',
      color: 'var(--accent-yellow)',
    };
  }
  if (usage === 'full_qkv') {
    return {
      label: 'Full QKV',
      detail: 'Standard Q/K/V attention primitives are present.',
      color: 'var(--accent-blue)',
    };
  }
  const inferred = classifyTokenMixing(program).qkvFree;
  if (inferred === true) {
    return {
      label: 'QKV-free*',
      detail: 'Inferred from graph ops when qkv_usage enum is unavailable.',
      color: 'var(--accent-green)',
    };
  }
  if (inferred === false) {
    return {
      label: 'Uses QKV*',
      detail: 'Inferred from graph ops when qkv_usage enum is unavailable.',
      color: 'var(--accent-yellow)',
    };
  }
  return {
    label: 'QKV unknown',
    detail: 'Insufficient graph/payload info to classify QKV usage.',
    color: 'var(--text-muted)',
  };
}

function AlternativesToAttention({ programs }) {
  const analysis = useMemo(() => {
    const familyStats = {};
    let qkvFreeCount = 0;
    let qkvCount = 0;
    let unknownCount = 0;
    const familyPrograms = {};

    for (const p of programs) {
      const { families, qkvFree } = classifyTokenMixing(p);
      if (qkvFree === null) { unknownCount++; continue; }
      if (qkvFree) qkvFreeCount++;
      else qkvCount++;

      for (const fam of families) {
        if (!familyStats[fam]) {
          familyStats[fam] = { count: 0, totalLoss: 0, totalNovelty: 0, bestLoss: Infinity, bestFingerprint: null };
          familyPrograms[fam] = [];
        }
        familyStats[fam].count++;
        if (p.loss_ratio != null) familyStats[fam].totalLoss += p.loss_ratio;
        if (p.novelty_score != null) familyStats[fam].totalNovelty += p.novelty_score;
        if (p.loss_ratio != null && p.loss_ratio < familyStats[fam].bestLoss) {
          familyStats[fam].bestLoss = p.loss_ratio;
          familyStats[fam].bestFingerprint = (p.graph_fingerprint || '').slice(0, 12);
        }
        familyPrograms[fam].push(p);
      }
    }

    const sorted = Object.entries(familyStats)
      .map(([fam, stats]) => ({
        family: fam,
        ...stats,
        avgLoss: stats.count > 0 ? stats.totalLoss / stats.count : null,
        avgNovelty: stats.count > 0 ? stats.totalNovelty / stats.count : null,
      }))
      .sort((a, b) => b.count - a.count);

    return { sorted, qkvFreeCount, qkvCount, unknownCount, total: programs.length };
  }, [programs]);

  if (analysis.sorted.length === 0) return null;

  return (
    <div className="card">
      <div className="card-title">Alternatives to Attention</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Token mixing mechanism breakdown across top programs. Shows which non-attention mechanisms
        appear in surviving architectures and their relative performance.
      </p>

      <div style={{ display: 'flex', gap: 16, marginBottom: 16, flexWrap: 'wrap' }}>
        <div style={{
          padding: '8px 14px', borderRadius: 6, background: 'var(--bg-tertiary)',
          borderLeft: '3px solid var(--accent-green)',
        }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-green)' }}>
            {analysis.qkvFreeCount}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>QKV-free</div>
        </div>
        <div style={{
          padding: '8px 14px', borderRadius: 6, background: 'var(--bg-tertiary)',
          borderLeft: '3px solid var(--accent-blue)',
        }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-blue)' }}>
            {analysis.qkvCount}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Uses QKV</div>
        </div>
        {analysis.unknownCount > 0 && (
          <div style={{
            padding: '8px 14px', borderRadius: 6, background: 'var(--bg-tertiary)',
            borderLeft: '3px solid var(--text-muted)',
          }}>
            <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-muted)' }}>
              {analysis.unknownCount}
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Unknown</div>
          </div>
        )}
      </div>

      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
            <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Mechanism</th>
            <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Programs</th>
            <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Avg Loss</th>
            <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Avg Novelty</th>
            <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Best (Loss)</th>
          </tr>
        </thead>
        <tbody>
          {analysis.sorted.map(row => (
            <tr key={row.family} style={{ borderBottom: '1px solid var(--border)' }}>
              <td style={{ padding: '6px 8px' }}>
                <span style={{
                  display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
                  background: FAMILY_COLORS[row.family] || 'var(--text-muted)',
                  marginRight: 6,
                }} />
                {FAMILY_LABELS[row.family] || row.family}
              </td>
              <td style={{ padding: '6px 8px', fontWeight: 600 }}>{row.count}</td>
              <td style={{
                padding: '6px 8px',
                color: row.avgLoss != null && row.avgLoss < 0.6 ? 'var(--accent-green)' : 'var(--text-secondary)',
              }}>
                {row.avgLoss != null ? row.avgLoss.toFixed(4) : '--'}
              </td>
              <td style={{
                padding: '6px 8px',
                color: row.avgNovelty != null && row.avgNovelty > 0.5 ? 'var(--accent-green)' : 'var(--text-secondary)',
              }}>
                {row.avgNovelty != null ? row.avgNovelty.toFixed(3) : '--'}
              </td>
              <td style={{ padding: '6px 8px', fontFamily: 'monospace', fontSize: 11 }}>
                {row.bestLoss < Infinity ? (
                  <span title={`Best: ${row.bestFingerprint}`}>
                    {row.bestLoss.toFixed(4)} ({row.bestFingerprint})
                  </span>
                ) : '--'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
        A program can use multiple mechanisms (e.g., conv + SSM). QKV-free means no attention primitives
        (local_window_attn, sliding_window_mask, multi_head_mix) are present in the graph.
      </div>
      <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
        Per-candidate tags use: <strong>Full QKV</strong> (standard attention), <strong>Q=K=V</strong> (shared-projection variant),
        and <strong>QKV-free</strong> (non-attention token mixing).
      </div>
    </div>
  );
}

function FunctionalFamilyEvidence({ coverage }) {
  const families = Array.isArray(coverage?.families) ? coverage.families : [];
  const totals = coverage?.totals || {};
  if (families.length === 0) return null;

  const functional = families.find(row => row.family === 'functional') || null;
  const exoticFamilies = families.filter(row => row.family !== 'euclidean');
  const exoticTested = exoticFamilies.reduce((sum, row) => sum + (row.n_tested || 0), 0);
  const exoticSurvived = exoticFamilies.reduce((sum, row) => sum + (row.n_survived || 0), 0);
  const testedBand = reliabilityBand(functional?.n_tested || 0);
  const survivedBand = reliabilityBand(functional?.n_survived || 0);

  return (
    <div className="card">
      <div className="card-title">Functional-Family Search Coverage</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Decision-focused evidence of whether exotic mathematical families, especially functional operators,
        are actually being explored and surviving stage-1 checks.
      </p>

      <div style={{ display: 'flex', gap: 14, marginBottom: 14, flexWrap: 'wrap' }}>
        <div style={{ padding: '8px 12px', borderRadius: 6, background: 'var(--bg-tertiary)', borderLeft: '3px solid var(--accent-purple)' }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-purple)' }}>{totals.n_tested || 0}</div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Total Tested</div>
        </div>
        <div style={{ padding: '8px 12px', borderRadius: 6, background: 'var(--bg-tertiary)', borderLeft: '3px solid var(--accent-green)' }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-green)' }}>{totals.n_survived || 0}</div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Total Survivors</div>
        </div>
        <div style={{ padding: '8px 12px', borderRadius: 6, background: 'var(--bg-tertiary)', borderLeft: '3px solid var(--accent-yellow)' }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-yellow)' }}>{exoticTested}</div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Exotic Tested</div>
        </div>
      </div>

      {functional && (
        <div style={{ marginBottom: 12, fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.55 }}>
          <div>
            <strong>Functional family tested:</strong> {functional.n_tested} ({(functional.tested_share * 100).toFixed(1)}% of all programs)
            {' · '}
            <strong style={{ color: testedBand.color, textTransform: 'uppercase', fontSize: 10 }}>{testedBand.label} sample depth</strong>
          </div>
          <div>
            <strong>Functional survivors:</strong> {functional.n_survived} (S1 rate {(functional.survival_rate * 100).toFixed(1)}%)
            {' · '}
            <strong style={{ color: survivedBand.color, textTransform: 'uppercase', fontSize: 10 }}>{survivedBand.label} survivor evidence</strong>
          </div>
          <div>
            <strong>Exotic family survivors:</strong> {exoticSurvived} across hyperbolic/tropical/p-adic/clifford/functional.
          </div>
        </div>
      )}

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Family</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Tested</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Survived</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>S1 Rate</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Share of Tested</th>
            </tr>
          </thead>
          <tbody>
            {families.map(row => {
              const isFunctional = row.family === 'functional';
              const testedShare = Number(row.tested_share || 0) * 100;
              const survivalRate = Number(row.survival_rate || 0) * 100;
              return (
                <tr key={row.family} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '6px 8px', fontWeight: isFunctional ? 700 : 500, color: isFunctional ? 'var(--accent-purple)' : 'var(--text-secondary)' }}>
                    {row.family}
                  </td>
                  <td style={{ padding: '6px 8px' }}>{row.n_tested}</td>
                  <td style={{ padding: '6px 8px' }}>{row.n_survived}</td>
                  <td style={{ padding: '6px 8px', color: survivalRate >= 10 ? 'var(--accent-green)' : 'var(--text-secondary)' }}>
                    {survivalRate.toFixed(1)}%
                  </td>
                  <td style={{ padding: '6px 8px' }}>{testedShare.toFixed(1)}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function MathspaceOperatorImpact({ impact }) {
  const rows = Array.isArray(impact?.by_operator) ? impact.by_operator : [];
  const families = Array.isArray(impact?.by_family) ? impact.by_family : [];
  const topTrust = Array.isArray(impact?.top_trustworthy_operators) ? impact.top_trustworthy_operators : [];
  const totals = impact?.totals || {};

  if (!impact || impact.available === false || rows.length === 0) {
    return null;
  }

  return (
    <div className="card">
      <div className="card-title">Mathspace Operator Impact</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Which mathspace operators are most represented and how they correlate with Stage-1/validation outcomes.
      </p>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 10 }}>
        <strong style={{ color: 'var(--accent-purple)' }}>Coverage:</strong>{' '}
        {totals.n_programs_with_mathspace ?? 0}/{totals.n_programs_with_graph ?? 0} programs with graph traces use mathspace ops.
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Trust score = (50% S1 pass + 30% validation pass + 20% baseline wins) × sample reliability,
        where sample reliability scales with tested count up to 25 programs.
      </div>

      {topTrust.length > 0 && (
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
          {topTrust.map(row => (
            <span
              key={row.op_name}
              style={{
                fontSize: 11,
                padding: '4px 8px',
                borderRadius: 999,
                border: `1px solid ${row.trust_label === 'high' ? 'var(--accent-green)' : row.trust_label === 'medium' ? 'var(--accent-yellow)' : 'var(--text-muted)'}`,
                color: row.trust_label === 'high' ? 'var(--accent-green)' : row.trust_label === 'medium' ? 'var(--accent-yellow)' : 'var(--text-muted)',
                background: 'var(--bg-tertiary)',
              }}
            >
              {row.op_name} · trust {(Number(row.trust_score || 0) * 100).toFixed(0)}%
            </span>
          ))}
        </div>
      )}

      <div style={{ overflowX: 'auto', marginBottom: 10 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
              <th style={{ padding: '6px 8px' }}>Operator</th>
              <th style={{ padding: '6px 8px' }}>N</th>
              <th style={{ padding: '6px 8px' }}>S1 %</th>
              <th style={{ padding: '6px 8px' }}>Validation %</th>
              <th style={{ padding: '6px 8px' }}>Baseline Win %</th>
              <th style={{ padding: '6px 8px' }}>Trust %</th>
              <th style={{ padding: '6px 8px' }}>Avg Novelty</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 12).map(row => (
              <tr key={row.op_name} style={{ borderBottom: '1px solid var(--border)' }}>
                <td style={{ padding: '6px 8px', color: 'var(--accent-blue)' }}>{row.op_name}</td>
                <td style={{ padding: '6px 8px' }}>{row.n_tested ?? 0}</td>
                <td style={{ padding: '6px 8px' }}>{((row.stage1_pass_rate || 0) * 100).toFixed(1)}%</td>
                <td style={{ padding: '6px 8px' }}>{((row.validation_pass_rate || 0) * 100).toFixed(1)}%</td>
                <td style={{ padding: '6px 8px' }}>{((row.baseline_win_rate || 0) * 100).toFixed(1)}%</td>
                <td style={{ padding: '6px 8px' }}>{((row.trust_score || 0) * 100).toFixed(1)}%</td>
                <td style={{ padding: '6px 8px' }}>{row.avg_novelty_score != null ? Number(row.avg_novelty_score).toFixed(3) : '--'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {families.length > 0 && (
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 11, color: 'var(--text-muted)' }}>
          {families.map(row => (
            <span key={row.family}>
              <strong style={{ color: 'var(--accent-purple)' }}>{row.family}:</strong> S1 {(row.stage1_pass_rate * 100).toFixed(0)}% · V {(row.validation_pass_rate * 100).toFixed(0)}%
            </span>
          ))}
        </div>
      )}

      {impact.explanation && (
        <div style={{ marginTop: 10, fontSize: 11, color: 'var(--text-muted)' }}>{impact.explanation}</div>
      )}
    </div>
  );
}

function RoutingModeComparison({ programs, comparison }) {
  const analysis = useMemo(() => {
    if (comparison && Array.isArray(comparison.by_mode)) {
      const sorted = [...comparison.by_mode]
        .map((row) => ({
          mode: row.routing_mode,
          count: row.n_programs || 0,
          sampleLabel: row.sample_size_label || 'unknown',
          confidenceLabel: row.confidence_label || 'unknown',
          stabilityLabel: row.stability_label || 'unknown',
          s1Rate: row.stage1_pass_rate || 0,
          avgLoss: row.avg_loss_ratio,
          avgDrop: row.avg_drop_rate,
          avgEntropy: row.avg_utilization_entropy,
          avgConf: row.avg_confidence_mean,
          tokenRetention: row.token_retention,
        }))
        .sort((a, b) => b.count - a.count);

      return {
        sorted,
        routedCount: comparison.routed_programs || 0,
        uniformCount: comparison.uniform_programs || 0,
        total: comparison.total_programs || 0,
      };
    }

    const byMode = {};
    let routedCount = 0;
    let uniformCount = 0;

    for (const p of programs) {
      const mode = p.routing_mode;
      if (!mode) { uniformCount++; continue; }
      routedCount++;
      if (!byMode[mode]) {
        byMode[mode] = {
          count: 0, s1Pass: 0, totalLoss: 0, lossCount: 0,
          totalDrop: 0, dropCount: 0, totalEntropy: 0, entropyCount: 0,
          totalConf: 0, confCount: 0, bestLoss: Infinity, bestFingerprint: null,
        };
      }
      const m = byMode[mode];
      m.count++;
      if (p.stage1_passed) m.s1Pass++;
      if (p.loss_ratio != null) { m.totalLoss += p.loss_ratio; m.lossCount++; }
      if (p.routing_drop_rate != null) { m.totalDrop += p.routing_drop_rate; m.dropCount++; }
      if (p.routing_utilization_entropy != null) { m.totalEntropy += p.routing_utilization_entropy; m.entropyCount++; }
      if (p.routing_confidence_mean != null) { m.totalConf += p.routing_confidence_mean; m.confCount++; }
      if (p.loss_ratio != null && p.loss_ratio < m.bestLoss) {
        m.bestLoss = p.loss_ratio;
        m.bestFingerprint = (p.graph_fingerprint || '').slice(0, 12);
      }
    }

    const sorted = Object.entries(byMode)
      .map(([mode, m]) => ({
        mode,
        count: m.count,
        sampleLabel: m.count >= 80 ? 'high' : m.count >= 30 ? 'medium' : 'low',
        confidenceLabel: 'unknown',
        stabilityLabel: 'unknown',
        s1Rate: m.count > 0 ? m.s1Pass / m.count : 0,
        avgLoss: m.lossCount > 0 ? m.totalLoss / m.lossCount : null,
        avgDrop: m.dropCount > 0 ? m.totalDrop / m.dropCount : null,
        avgEntropy: m.entropyCount > 0 ? m.totalEntropy / m.entropyCount : null,
        avgConf: m.confCount > 0 ? m.totalConf / m.confCount : null,
        bestLoss: m.bestLoss < Infinity ? m.bestLoss : null,
        bestFingerprint: m.bestFingerprint,
        tokenRetention: m.dropCount > 0 ? Math.max(0, 1 - (m.totalDrop / m.dropCount)) : null,
      }))
      .sort((a, b) => b.count - a.count);

    return { sorted, routedCount, uniformCount, total: programs.length };
  }, [programs]);

  if (analysis.sorted.length === 0) return null;

  return (
    <div className="card">
      <div className="card-title">Routing Mode Comparison</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Consolidated routing-mode evidence across uniform and routed candidates.
        Includes sample-size and confidence labels to avoid over-reading small-N differences.
      </p>

      <div style={{ display: 'flex', gap: 16, marginBottom: 16, flexWrap: 'wrap' }}>
        <div style={{
          padding: '8px 14px', borderRadius: 6, background: 'var(--bg-tertiary)',
          borderLeft: '3px solid var(--accent-purple)',
        }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-purple)' }}>
            {analysis.routedCount}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Routed</div>
        </div>
        <div style={{
          padding: '8px 14px', borderRadius: 6, background: 'var(--bg-tertiary)',
          borderLeft: '3px solid var(--text-muted)',
        }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-muted)' }}>
            {analysis.uniformCount}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Uniform (no routing)</div>
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Mode</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>N</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Sample</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>S1 Rate</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Avg Loss</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Drop %</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Entropy</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Confidence</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Conf Label</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Stability</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Token Retention</th>
            </tr>
          </thead>
          <tbody>
            {analysis.sorted.map(row => (
              <tr key={row.mode} style={{ borderBottom: '1px solid var(--border)' }}>
                <td style={{ padding: '6px 8px', color: 'var(--accent-blue)', fontWeight: 600 }}>{row.mode}</td>
                <td style={{ padding: '6px 8px' }}>{row.count}</td>
                <td style={{ padding: '6px 8px', textTransform: 'uppercase', fontSize: 11 }}>{row.sampleLabel}</td>
                <td style={{
                  padding: '6px 8px',
                  color: row.s1Rate > 0.5 ? 'var(--accent-green)' : row.s1Rate > 0.2 ? 'var(--accent-yellow)' : 'var(--text-secondary)',
                }}>
                  {(row.s1Rate * 100).toFixed(0)}%
                </td>
                <td style={{
                  padding: '6px 8px',
                  color: row.avgLoss != null && row.avgLoss < 0.6 ? 'var(--accent-green)' : 'var(--text-secondary)',
                }}>
                  {row.avgLoss != null ? row.avgLoss.toFixed(4) : '--'}
                </td>
                <td style={{
                  padding: '6px 8px',
                  color: row.avgDrop != null
                    ? (row.avgDrop > 0.3 ? 'var(--accent-red)' : row.avgDrop > 0.1 ? 'var(--accent-yellow)' : 'var(--accent-green)')
                    : 'var(--text-muted)',
                }}>
                  {row.avgDrop != null ? `${(row.avgDrop * 100).toFixed(1)}%` : '--'}
                </td>
                <td style={{ padding: '6px 8px', color: 'var(--text-secondary)' }}>
                  {row.avgEntropy != null ? row.avgEntropy.toFixed(3) : '--'}
                </td>
                <td style={{
                  padding: '6px 8px',
                  color: row.avgConf != null
                    ? (row.avgConf > 0.8 ? 'var(--accent-green)' : row.avgConf > 0.5 ? 'var(--accent-yellow)' : 'var(--accent-red)')
                    : 'var(--text-muted)',
                }}>
                  {row.avgConf != null ? row.avgConf.toFixed(3) : '--'}
                </td>
                <td style={{ padding: '6px 8px', textTransform: 'uppercase', fontSize: 11 }}>{row.confidenceLabel}</td>
                <td style={{ padding: '6px 8px', textTransform: 'uppercase', fontSize: 11 }}>{row.stabilityLabel}</td>
                <td style={{ padding: '6px 8px' }}>{row.tokenRetention != null ? `${(row.tokenRetention * 100).toFixed(1)}%` : '--'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
        Sample labels reflect evidence depth by mode (`high`, `medium`, `low`).
        Confidence labels combine confidence mean and variance; stability reflects confidence variance.
      </div>
    </div>
  );
}

const WEIGHT_STORAGE_LABELS = {
  dense_matrix: 'Dense (baseline)',
  low_rank: 'Low-Rank (UV)',
  hypernetwork: 'Hypernetwork',
  shared_basis: 'Shared Basis',
  hash_trick: 'Hash Trick',
  kronecker: 'Kronecker',
  polynomial: 'Polynomial',
  structured_sparse: 'Structured Sparse',
};

const TOKEN_REP_LABELS = {
  standard_float: 'Standard Float',
  binary_hash: 'Binary Hash',
  residual_quantized: 'Residual Quantized',
  complex_valued: 'Complex',
  quaternion: 'Quaternion',
  multi_resolution: 'Multi-Resolution',
  mixture_embedding: 'Mixture Embedding',
};

function CompressionTechniqueCoverage({ programs }) {
  const analysis = useMemo(() => {
    const byTechnique = {};
    let denseCount = 0;
    let compressedCount = 0;

    for (const p of programs) {
      const spec = parseArchSpec(p.arch_spec_json);
      const ws = spec?.choices?.weight_storage || 'dense_matrix';
      const tr = spec?.choices?.token_representation;
      const isDense = ws === 'dense_matrix' && (!tr || tr === 'standard_float');

      if (isDense) { denseCount++; } else { compressedCount++; }

      // Track weight storage technique
      const key = ws !== 'dense_matrix' ? ws : (tr && tr !== 'standard_float' ? tr : 'dense_matrix');
      if (!byTechnique[key]) {
        byTechnique[key] = {
          count: 0, s1Pass: 0, totalLoss: 0, lossCount: 0,
          totalParams: 0, paramsCount: 0, bestLoss: Infinity, bestFingerprint: null,
        };
      }
      const m = byTechnique[key];
      m.count++;
      if (p.stage1_passed) m.s1Pass++;
      if (p.loss_ratio != null) { m.totalLoss += p.loss_ratio; m.lossCount++; }
      if (p.param_count != null) { m.totalParams += p.param_count; m.paramsCount++; }
      if (p.loss_ratio != null && p.loss_ratio < m.bestLoss) {
        m.bestLoss = p.loss_ratio;
        m.bestFingerprint = (p.graph_fingerprint || '').slice(0, 12);
      }
    }

    const sorted = Object.entries(byTechnique)
      .map(([technique, m]) => ({
        technique,
        label: WEIGHT_STORAGE_LABELS[technique] || TOKEN_REP_LABELS[technique] || technique,
        count: m.count,
        s1Rate: m.count > 0 ? m.s1Pass / m.count : 0,
        avgLoss: m.lossCount > 0 ? m.totalLoss / m.lossCount : null,
        avgParams: m.paramsCount > 0 ? m.totalParams / m.paramsCount : null,
        factor: COMPRESSION_FACTORS[technique] || 1.0,
        bestLoss: m.bestLoss < Infinity ? m.bestLoss : null,
        bestFingerprint: m.bestFingerprint,
      }))
      .sort((a, b) => b.count - a.count);

    return { sorted, denseCount, compressedCount, total: programs.length };
  }, [programs]);

  // Only show if there's at least one non-dense technique
  if (analysis.compressedCount === 0) return null;

  return (
    <div className="card">
      <div className="card-title">Compression Technique Coverage</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Weight storage and token representation techniques used across stage-1 survivors.
        Compressed architectures use fewer parameters for comparable or better performance.
      </p>

      <div style={{ display: 'flex', gap: 16, marginBottom: 16, flexWrap: 'wrap' }}>
        <div style={{
          padding: '8px 14px', borderRadius: 6, background: 'var(--bg-tertiary)',
          borderLeft: '3px solid var(--accent-green)',
        }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-green)' }}>
            {analysis.compressedCount}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Compressed</div>
        </div>
        <div style={{
          padding: '8px 14px', borderRadius: 6, background: 'var(--bg-tertiary)',
          borderLeft: '3px solid var(--text-muted)',
        }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-muted)' }}>
            {analysis.denseCount}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Dense (baseline)</div>
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Technique</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>N</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>S1 Rate</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Avg Loss</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Avg Params</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Est. Ratio</th>
              <th style={{ padding: '6px 8px', color: 'var(--text-muted)', fontSize: 11 }}>Best (Loss)</th>
            </tr>
          </thead>
          <tbody>
            {analysis.sorted.map(row => (
              <tr key={row.technique} style={{ borderBottom: '1px solid var(--border)' }}>
                <td style={{ padding: '6px 8px', fontWeight: 600, color: row.factor < 1 ? 'var(--accent-green)' : 'var(--text-secondary)' }}>
                  {row.label}
                </td>
                <td style={{ padding: '6px 8px' }}>{row.count}</td>
                <td style={{
                  padding: '6px 8px',
                  color: row.s1Rate > 0.5 ? 'var(--accent-green)' : row.s1Rate > 0.2 ? 'var(--accent-yellow)' : 'var(--text-secondary)',
                }}>
                  {(row.s1Rate * 100).toFixed(0)}%
                </td>
                <td style={{
                  padding: '6px 8px',
                  color: row.avgLoss != null && row.avgLoss < 0.6 ? 'var(--accent-green)' : 'var(--text-secondary)',
                }}>
                  {row.avgLoss != null ? row.avgLoss.toFixed(4) : '--'}
                </td>
                <td style={{ padding: '6px 8px', color: 'var(--text-secondary)' }}>
                  {row.avgParams != null ? `${(row.avgParams / 1e6).toFixed(2)}M` : '--'}
                </td>
                <td style={{
                  padding: '6px 8px',
                  color: row.factor < 1 ? 'var(--accent-green)' : 'var(--text-muted)',
                }}>
                  {row.factor < 1 ? `${(row.factor * 100).toFixed(0)}%` : '100%'}
                </td>
                <td style={{ padding: '6px 8px', fontFamily: 'monospace', fontSize: 11 }}>
                  {row.bestLoss != null ? (
                    <span title={`Best: ${row.bestFingerprint}`}>
                      {row.bestLoss.toFixed(4)} ({row.bestFingerprint})
                    </span>
                  ) : '--'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
        Est. Ratio = estimated parameter retention after compression (lower = more compressed).
        Techniques from the morphological box weight_storage and token_representation dimensions.
      </div>
    </div>
  );
}

function RatingBadge({ program }) {
  const lr = program.loss_ratio;
  const nov = program.novelty_score || 0;
  const bl = program.baseline_loss_ratio;

  let color, label;
  if (bl != null && bl < 1 && lr < 0.5 && nov > 0.7) {
    color = 'var(--accent-green)'; label = 'S1 - Exceptional';
  } else if (lr < 0.5 && nov > 0.5) {
    color = 'var(--accent-green)'; label = 'S1 - Strong';
  } else if (lr < 0.7) {
    color = 'var(--accent-yellow)'; label = 'S1 - Moderate';
  } else {
    color = 'var(--accent-orange, #f0883e)'; label = 'S1 - Marginal';
  }

  return (
    <span style={{
      padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600,
      background: `${color}22`, color, border: `1px solid ${color}44`,
    }}>
      {label}
    </span>
  );
}

function reliabilityBand(sampleSize) {
  if (sampleSize >= 30) return { label: 'high', color: 'var(--accent-green)' };
  if (sampleSize >= 12) return { label: 'medium', color: 'var(--accent-yellow)' };
  return { label: 'low', color: 'var(--accent-red)' };
}

function wilsonInterval(successes, total, z = 1.96) {
  if (!Number.isFinite(successes) || !Number.isFinite(total) || total <= 0) {
    return null;
  }
  const p = successes / total;
  const z2 = z * z;
  const denom = 1 + z2 / total;
  const center = p + z2 / (2 * total);
  const margin = z * Math.sqrt((p * (1 - p) + z2 / (4 * total)) / total);
  const low = Math.max(0, (center - margin) / denom);
  const high = Math.min(1, (center + margin) / denom);
  return { low, high };
}

function promotionEvidence(program) {
  const seenRuns = Number(program?.cross_run_stability?.seen_runs || 0);
  const baselineRatioValue = Number(program?.baseline_loss_ratio);
  const stdValue = Number(program?.validation_multi_seed_std);
  const baselineRatio = Number.isFinite(baselineRatioValue) ? baselineRatioValue : null;
  const std = Number.isFinite(stdValue) ? stdValue : null;
  const checks = {
    lossEvidence: program?.loss_ratio != null,
    noveltyEvidence: program?.novelty_score != null,
    baselineEvidence: baselineRatio != null,
    baselineBeat: baselineRatio != null && baselineRatio < 1.0,
    ckaArtifactBacked: program?.cka_source === 'artifact',
    repeatObserved: seenRuns >= 3,
    multiSeedStd: std != null,
    boundedStd: std != null && std <= 0.12,
  };
  const totalChecks = Object.keys(checks).length;
  const evidenceCount = Object.values(checks).filter(Boolean).length;
  const completeness = evidenceCount / totalChecks;
  const stdSignal = std == null ? 0 : std <= 0.05 ? 1 : std <= 0.12 ? 0.65 : std <= 0.2 ? 0.35 : 0.1;
  const repeatSignal = seenRuns >= 5 ? 1 : seenRuns >= 3 ? 0.65 : seenRuns >= 2 ? 0.4 : seenRuns >= 1 ? 0.2 : 0;
  const margin = baselineRatio == null ? null : 1 - baselineRatio;
  const marginSignal = margin == null ? 0 : margin >= 0.1 ? 1 : margin > 0 ? 0.7 : 0.15;
  const score = Math.round((completeness * 0.55 + stdSignal * 0.15 + repeatSignal * 0.2 + marginSignal * 0.1) * 100);
  const confidence = score >= 75
    ? { label: 'High', color: 'var(--accent-green)' }
    : score >= 45
      ? { label: 'Moderate', color: 'var(--accent-yellow)' }
      : { label: 'Low', color: 'var(--accent-red)' };
  const uncertaintyLabel = std == null
    ? 'unknown'
    : std <= 0.05 ? 'tight'
      : std <= 0.12 ? 'bounded'
        : 'high';
  return {
    ...confidence,
    score,
    seenRuns,
    std,
    uncertaintyLabel,
    evidenceCount,
    totalChecks,
  };
}

function reproducibilityPacketStatus(program) {
  const spec = parseArchSpec(program?.arch_spec_json);
  const checks = [
    { label: 'result_id', ok: !!program?.result_id },
    { label: 'graph_fingerprint', ok: !!program?.graph_fingerprint },
    { label: 'arch_spec', ok: !!spec },
    { label: 'loss_ratio', ok: program?.loss_ratio != null },
    { label: 'baseline_ratio', ok: program?.baseline_loss_ratio != null },
    { label: 'cka_artifact', ok: program?.cka_source === 'artifact' },
  ];
  const readyCount = checks.filter(check => check.ok).length;
  const totalChecks = checks.length;
  const label = readyCount === totalChecks ? 'Ready' : readyCount >= 4 ? 'Partial' : 'Sparse';
  const color = readyCount === totalChecks
    ? 'var(--accent-green)'
    : readyCount >= 4
      ? 'var(--accent-yellow)'
      : 'var(--accent-red)';
  return {
    label,
    color,
    readyCount,
    totalChecks,
  };
}

function decisionGate(program) {
  const checks = {
    screeningEvidence: program.loss_ratio != null && program.novelty_score != null,
    baselineEvidence: program.baseline_loss_ratio != null,
    baselineBeatsReference: program.baseline_loss_ratio != null && program.baseline_loss_ratio < 1.0,
    ckaArtifactBacked: program.cka_source === 'artifact',
  };
  const decisionReady = Object.values(checks).every(Boolean);
  const missing = Object.entries(checks)
    .filter(([, ok]) => !ok)
    .map(([name]) => name);
  return {
    decisionReady,
    label: decisionReady ? 'Decision-Ready' : 'Exploratory',
    color: decisionReady ? 'var(--accent-green)' : 'var(--accent-yellow)',
    missing,
  };
}

const DISC_COLUMNS = [
  { key: '_score', label: 'Score' },
  { key: 'graph_fingerprint', label: 'Fingerprint' },
  { key: 'repeat_count', label: 'Repeats' },
  { key: 'loss_ratio', label: 'Loss Ratio' },
  { key: 'novelty_score', label: 'Novelty' },
  { key: 'baseline_loss_ratio', label: 'Baseline' },
  { key: '_compressionRatio', label: 'Compression' },
  { key: '_metricQualityOrder', label: 'Metric Quality' },
  { key: 'cka_source', label: 'CKA Source' },
  { key: 'most_similar_to', label: 'Similar To' },
  { key: '_decisionGateOrder', label: 'Decision Gate' },
  { key: 'rating', label: 'Rating' },
];

const REPORT_DISCOVERY_SORT_PREFS_KEY = 'dashboard.report.discovery-rankings.sort.v1';
const REPORT_DISCOVERY_VIEW_PREFS_KEY = 'dashboard.report.discovery-rankings.view.v1';

const DISC_RATING_ORDER = { 'S1 - Exceptional': 4, 'S1 - Strong': 3, 'S1 - Moderate': 2, 'S1 - Marginal': 1 };

function reportQueueReasonLabel(reason) {
  if (reason === 'already_investigated_unchanged') return 'Already investigated (unchanged).';
  if (reason === 'not_investigation_passed') return 'Investigation did not pass robustness gate.';
  if (reason === 'already_validated') return 'Already validated.';
  if (reason === 'not_investigation_tier') return 'Validation requires investigation tier.';
  if (reason === 'not_screening_tier') return 'Investigation requires screening tier.';
  if (reason === 'not_stage1_survivor') return 'Candidate is not a Stage-1 survivor.';
  if (reason === 'not_in_leaderboard') return 'Candidate is not in progression leaderboard yet.';
  if (reason === 'result_not_found') return 'Result ID not found.';
  return 'Candidate is not currently eligible for progression actions.';
}

function DiscoveryRankings({ programs, expandedPrograms, onSelectProgram, onInvestigate, onValidate, onQueueAdd, onQueueRemove, queuedResultIds, eligibilityByResultId }) {
  const [viewMode, setViewMode] = useState(() => {
    try {
      const stored = localStorage.getItem(REPORT_DISCOVERY_VIEW_PREFS_KEY);
      return stored === 'expanded' ? 'expanded' : 'grouped';
    } catch {}
    return 'grouped';
  });
  const [sortKey, setSortKey] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(REPORT_DISCOVERY_SORT_PREFS_KEY) || '{}');
      const validKeys = new Set([...DISC_COLUMNS.map((column) => column.key), '_ratingOrder']);
      if (typeof stored.sortKey === 'string' && validKeys.has(stored.sortKey)) {
        return stored.sortKey;
      }
    } catch {}
    return '_score';
  });
  const [sortDesc, setSortDesc] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(REPORT_DISCOVERY_SORT_PREFS_KEY) || '{}');
      if (typeof stored.sortDesc === 'boolean') {
        return stored.sortDesc;
      }
    } catch {}
    return true;
  });
  const [copiedValue, copyText] = useCopyToClipboard();
  const queuedSet = useMemo(() => new Set(queuedResultIds || []), [queuedResultIds]);

  useEffect(() => {
    try {
      localStorage.setItem(REPORT_DISCOVERY_SORT_PREFS_KEY, JSON.stringify({ sortKey, sortDesc }));
    } catch {}
  }, [sortKey, sortDesc]);

  useEffect(() => {
    try {
      localStorage.setItem(REPORT_DISCOVERY_VIEW_PREFS_KEY, viewMode);
    } catch {}
  }, [viewMode]);

  const groupedRows = Array.isArray(programs) ? programs : [];
  const expandedRows = Array.isArray(expandedPrograms) && expandedPrograms.length > 0
    ? expandedPrograms
    : groupedRows;
  const isExpanded = viewMode === 'expanded';
  const sourceRows = isExpanded ? expandedRows : groupedRows;

  const groupedUnique = groupedRows.length;
  const expandedTotal = expandedRows.length;
  const rerunRows = expandedRows.filter(p => Number(p.group_repeat_count || p.repeat_count || 1) > 1).length;
  const rerunRatio = expandedTotal > 0 ? Math.round((rerunRows / expandedTotal) * 100) : 0;

  const sortAriaValue = (columnKey) => {
    const normalized = columnKey === 'rating' ? '_ratingOrder' : columnKey;
    if (sortKey !== normalized) return 'none';
    return sortDesc ? 'descending' : 'ascending';
  };

  const handleSort = (key) => {
    if (key === 'rating') key = '_ratingOrder';
    if (sortKey === key) setSortDesc(!sortDesc);
    else { setSortKey(key); setSortDesc(true); }
  };

  const sorted = useMemo(() => {
    const aug = sourceRows.map(p => {
      const repeatCount = Number(p.repeat_count || p.group_repeat_count || 1);
      const repeatIndex = Number(p.group_repeat_index || 1);
      const lr = p.loss_ratio;
      const nov = p.novelty_score || 0;
      const bl = p.baseline_loss_ratio;
      let rLabel;
      if (bl != null && bl < 1 && lr < 0.5 && nov > 0.7) rLabel = 'S1 - Exceptional';
      else if (lr < 0.5 && nov > 0.5) rLabel = 'S1 - Strong';
      else if (lr < 0.7) rLabel = 'S1 - Moderate';
      else rLabel = 'S1 - Marginal';
      const gate = decisionGate(p);
      const compression = compressionSummary(p);
      const chips = metricChips(p);
      const promotion = promotionEvidence(p);
      const reproPacket = reproducibilityPacketStatus(p);
      const qkv = qkvUsageDescriptor(p);
      return {
        ...p,
        repeat_count: repeatCount,
        group_repeat_count: repeatCount,
        group_repeat_index: repeatIndex,
        _score: discoveryScore(p),
        _scoreBreakdown: discoveryScoreBreakdown(p),
        _ratingOrder: DISC_RATING_ORDER[rLabel] || 0,
        _compressionRatio: compression.ratio ?? -1,
        _compressionSummary: compression,
        _metricQuality: chips,
        _metricQualityOrder: chips.filter(ch => ch.reliability === 'high').length,
        _decisionGateOrder: gate.decisionReady ? 1 : 0,
        _promotionEvidence: promotion,
        _reproPacket: reproPacket,
        _qkvDescriptor: qkv,
      };
    });
    aug.sort((a, b) => {
      let va, vb;
      if (sortKey === 'graph_fingerprint' || sortKey === 'most_similar_to') {
        va = a[sortKey] || ''; vb = b[sortKey] || '';
        return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      }
      va = a[sortKey]; vb = b[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return sortDesc ? vb - va : va - vb;
    });
    return aug;
  }, [sourceRows, sortKey, sortDesc]);

  return (
    <div className="card">
      <div className="card-title">Discovery Rankings</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        The strongest architectures discovered, ranked by a composite of learning speed, novelty, and baseline comparison.
        Higher score is better and is meant for triage (not a publication-grade metric).
      </p>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Rankings are fingerprint-deduplicated: each row is one architecture identity (`graph_fingerprint`) with repeat/run-spread metadata.
      </p>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 8,
        flexWrap: 'wrap',
        marginBottom: 12,
      }}>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
          Same architecture repeated means reruns of one fingerprint. Grouped view shows one representative per fingerprint; expanded view shows every rerun row.
          {expandedTotal > 0 && (
            <span> Current mix: {groupedUnique} unique architectures across {expandedTotal} rows ({rerunRatio}% reruns).</span>
          )}
        </div>
        <div style={{ display: 'inline-flex', gap: 6 }}>
          <button
            className="refresh-btn"
            style={{ fontSize: 11, padding: '4px 8px', opacity: isExpanded ? 0.8 : 1 }}
            onClick={() => setViewMode('grouped')}
            aria-pressed={!isExpanded}
          >
            Grouped view
          </button>
          <button
            className="refresh-btn"
            style={{ fontSize: 11, padding: '4px 8px', opacity: isExpanded ? 1 : 0.8 }}
            onClick={() => setViewMode('expanded')}
            aria-pressed={isExpanded}
          >
            Expanded reruns
          </button>
        </div>
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        <strong>Score bands:</strong> 70+ strong follow-up, 40-69 promising, below 40 low priority. Click a fingerprint to open full program detail.
      </p>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Decision gate: mark as <strong>Decision-Ready</strong> only when screening metrics are present, baseline ratio is {'<'} 1.00, and CKA source is artifact-backed.
      </p>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Compression column shows estimated parameter ratio vs dense baseline and memory footprint; Metric Quality chips show provenance (`artifact-backed`/`heuristic`) and reliability.
      </p>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        QKV alternative labels are candidate-level: <strong>Full QKV</strong>, <strong>Q=K=V</strong>, or <strong>QKV-free</strong>.
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
              <th scope="col" style={{ padding: '8px 6px', color: 'var(--text-muted)' }}>#</th>
              {DISC_COLUMNS.map(col => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  scope="col"
                  aria-sort={sortAriaValue(col.key)}
                  aria-label={`Sort by ${col.label}`}
                  style={{ padding: '8px 6px', color: 'var(--text-muted)', cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap' }}
                >
                  {col.label}
                  {(sortKey === col.key || (col.key === 'rating' && sortKey === '_ratingOrder')) && (
                    <span style={{ marginLeft: 4, fontSize: 10 }}>
                      {sortDesc ? '\u25BC' : '\u25B2'}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((p, i) => {
              const gate = decisionGate(p);
              const eligibility = (p.result_id && eligibilityByResultId?.[p.result_id]) || {
                investigationEligible: false,
                validationEligible: false,
                queueEligible: false,
                queueReason: 'not_progression_eligible',
              };
              const queueIntent = eligibility.validationEligible
                ? 'validation'
                : eligibility.investigationEligible
                  ? 'investigation'
                  : null;
              return (
              <tr key={p.result_id || i} style={{ borderBottom: '1px solid var(--border)' }}>
                <td style={{ padding: '6px', color: 'var(--text-muted)' }}>{i + 1}</td>
                <td style={{ padding: '6px', fontWeight: 600, color: scoreColor(p._score) }}>
                  <span title={`Loss ${(p._scoreBreakdown.loss || 0).toFixed(1)}/35 | Novelty ${(p._scoreBreakdown.novelty || 0).toFixed(1)}/25 | Baseline ${(p._scoreBreakdown.baseline || 0).toFixed(1)}/30 | ID ${(p._scoreBreakdown.id || 0).toFixed(1)}/10`}>
                    {p._score}
                  </span>
                </td>
                <td style={{ padding: '6px' }}>
                  {p.result_id && onSelectProgram ? (
                    <>
                      <button
                        className="refresh-btn"
                        style={{ fontSize: 11, padding: '3px 8px', fontFamily: 'monospace' }}
                        onClick={() => onSelectProgram(p.result_id)}
                        aria-label={`Open program details for fingerprint ${(p.graph_fingerprint || '').slice(0, 12)}`}
                      >
                        {(p.graph_fingerprint || '').slice(0, 12)}
                      </button>
                      {p.graph_fingerprint && (
                        <button
                          className="refresh-btn"
                          style={{ fontSize: 10, padding: '1px 5px', marginLeft: 6 }}
                          onClick={() => copyText(p.graph_fingerprint)}
                          aria-label={`Copy fingerprint ${p.graph_fingerprint}`}
                        >
                          {copiedValue === p.graph_fingerprint ? 'Copied FP' : 'Copy FP'}
                        </button>
                      )}
                      {p.result_id && (
                        <button
                          className="refresh-btn"
                          style={{ fontSize: 10, padding: '1px 5px', marginLeft: 4 }}
                          onClick={() => copyText(p.result_id)}
                          aria-label={`Copy result id ${p.result_id}`}
                        >
                          {copiedValue === p.result_id ? 'Copied ID' : 'Copy ID'}
                        </button>
                      )}
                    </>
                  ) : (
                    <span style={{ fontFamily: 'monospace', color: 'var(--accent-blue)' }}>
                      {(p.graph_fingerprint || '').slice(0, 12)}
                    </span>
                  )}
                  {(p.repeat_count || 1) > 1 && (
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
                      {isExpanded
                        ? `rerun ${p.group_repeat_index || 1} of ${p.group_repeat_count || p.repeat_count || 1}`
                        : `repeated ${p.repeat_count}x across ${p.repeat_experiment_span || 1} run${(p.repeat_experiment_span || 1) === 1 ? '' : 's'}`}
                    </div>
                  )}
                  {(p.repeat_loss_min != null || p.repeat_loss_max != null) && (
                    <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                      loss spread {p.repeat_loss_min != null ? p.repeat_loss_min.toFixed(4) : '--'} to {p.repeat_loss_max != null ? p.repeat_loss_max.toFixed(4) : '--'}
                    </div>
                  )}
                </td>
                <td style={{ padding: '6px' }}>
                  <span style={{ color: (p.repeat_count || 1) > 1 ? 'var(--accent-yellow)' : 'var(--text-muted)', fontWeight: 600 }}>
                    {p.repeat_count || 1}x
                  </span>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                    {isExpanded
                      ? `row ${p.group_repeat_index || 1} in fingerprint group`
                      : `span ${p.repeat_experiment_span || 1} run${(p.repeat_experiment_span || 1) === 1 ? '' : 's'}`}
                  </div>
                </td>
                <td style={{
                  padding: '6px', fontWeight: 600,
                  color: (p.loss_ratio || 1) < 0.5 ? 'var(--accent-green)' : (p.loss_ratio || 1) < 0.7 ? 'var(--accent-yellow)' : 'var(--text-secondary)',
                }}>
                  {p.loss_ratio != null ? p.loss_ratio.toFixed(4) : '--'}
                </td>
                <td style={{ padding: '6px', color: (p.novelty_score || 0) > 0.7 ? 'var(--accent-green)' : 'var(--text-secondary)' }}>
                  {p.novelty_score != null ? p.novelty_score.toFixed(3) : '--'}
                </td>
                <td style={{
                  padding: '6px',
                  color: p.baseline_loss_ratio != null && p.baseline_loss_ratio < 1 ? 'var(--accent-green)' : 'var(--text-secondary)',
                  fontWeight: p.baseline_loss_ratio != null && p.baseline_loss_ratio < 1 ? 600 : 'normal',
                }}>
                  {p.baseline_loss_ratio != null ? p.baseline_loss_ratio.toFixed(3) : '--'}
                </td>
                <td style={{ padding: '6px' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                    {p._compressionSummary?.ratio != null ? `${(p._compressionSummary.ratio * 100).toFixed(0)}%` : '--'}
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                    {p._compressionSummary?.memoryMb != null ? `${p._compressionSummary.memoryMb.toFixed(2)} MB` : 'n/a'} · {p._compressionSummary?.label || 'dense'}
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                    retention {p._compressionSummary?.qualityRetention != null ? `${(p._compressionSummary.qualityRetention * 100).toFixed(0)}%` : 'n/a'}
                  </div>
                </td>
                <td style={{ padding: '6px' }}>
                  <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', maxWidth: 220 }}>
                    {(p._metricQuality || []).map(chip => (
                      <span
                        key={`${p.result_id || i}-${chip.label}`}
                        title={`${chip.label}: ${chip.source}, ${chip.reliability} reliability`}
                        style={{
                          fontSize: 10,
                          padding: '1px 5px',
                          borderRadius: 4,
                          border: `1px solid ${reliabilityColor(chip.reliability)}55`,
                          color: reliabilityColor(chip.reliability),
                          background: `${reliabilityColor(chip.reliability)}22`,
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {chip.label}: {chip.source}
                      </span>
                    ))}
                  </div>
                  <div
                    style={{ marginTop: 5, fontSize: 10, fontWeight: 600, color: p._promotionEvidence?.color || 'var(--text-muted)' }}
                    title={`Evidence checks ${p._promotionEvidence?.evidenceCount || 0}/${p._promotionEvidence?.totalChecks || 0}`}
                  >
                    Promotion confidence: {p._promotionEvidence?.label || 'Low'} ({p._promotionEvidence?.score ?? 0}%)
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                    Uncertainty {p._promotionEvidence?.uncertaintyLabel || 'unknown'}; runs {p._promotionEvidence?.seenRuns || 0}; std {p._promotionEvidence?.std != null ? p._promotionEvidence.std.toFixed(3) : 'n/a'}
                  </div>
                  <div style={{ marginTop: 2, fontSize: 10, color: p._reproPacket?.color || 'var(--text-muted)' }}>
                    Repro packet: {p._reproPacket?.label || 'Sparse'} ({p._reproPacket?.readyCount || 0}/{p._reproPacket?.totalChecks || 0})
                  </div>
                </td>
                <td style={{ padding: '6px' }}>
                  {p.cka_source ? (
                    <span style={{
                      fontSize: 10,
                      fontWeight: 600,
                      padding: '2px 6px',
                      borderRadius: 4,
                      background: p.cka_source === 'artifact' ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)',
                      color: p.cka_source === 'artifact' ? 'var(--accent-green)' : 'var(--accent-red)',
                    }}>
                      {p.cka_source === 'artifact' ? 'artifact' : 'fallback'}
                    </span>
                  ) : '--'}
                  {p.cka_artifact_version && (
                    <span style={{ marginLeft: 6, fontSize: 10, color: 'var(--text-muted)' }}>
                      {p.cka_artifact_version}
                    </span>
                  )}
                </td>
                <td style={{ padding: '6px', color: 'var(--text-muted)', fontSize: 11 }}>
                  {p.most_similar_to || '--'}
                  <div
                    style={{ marginTop: 4, fontSize: 10, color: p._qkvDescriptor?.color || 'var(--text-muted)', fontWeight: 600 }}
                    title={p._qkvDescriptor?.detail || ''}
                  >
                    {p._qkvDescriptor?.label || 'QKV unknown'}
                  </div>
                </td>
                <td style={{ padding: '6px' }}>
                  <span
                    style={{
                      fontSize: 10,
                      fontWeight: 600,
                      textTransform: 'uppercase',
                      padding: '2px 6px',
                      borderRadius: 4,
                      color: gate.color,
                      background: `${gate.color}22`,
                      border: `1px solid ${gate.color}55`,
                    }}
                    title={gate.decisionReady
                      ? 'All report-level evidence checks passed.'
                      : `Missing checks: ${gate.missing.join(', ')}`}
                  >
                    {gate.label}
                  </span>
                </td>
                <td style={{ padding: '6px' }}>
                  {p.loss_ratio != null && <RatingBadge program={p} />}
                  {p.result_id && (
                    <div style={{ marginTop: 6, display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                      {onInvestigate && eligibility.investigationEligible && (
                        <button
                          className="refresh-btn"
                          style={{ fontSize: 10, padding: '1px 6px' }}
                          onClick={() => onInvestigate([p.result_id])}
                          aria-label={`Investigate program ${p.result_id}`}
                        >
                          Investigate
                        </button>
                      )}
                      {onValidate && eligibility.validationEligible && (
                        <button
                          className="refresh-btn"
                          style={{ fontSize: 10, padding: '1px 6px' }}
                          onClick={() => onValidate([p.result_id])}
                          aria-label={`Validate program ${p.result_id}`}
                        >
                          Validate
                        </button>
                      )}
                      {onSelectProgram && (
                        <button
                          className="refresh-btn"
                          style={{
                            fontSize: 10, padding: '1px 6px',
                            borderColor: 'var(--accent-purple)',
                            color: 'var(--accent-purple)',
                          }}
                          onClick={() => onSelectProgram(p.result_id)}
                          aria-label={`Open decision packet for ${p.result_id}`}
                        >
                          Packet
                        </button>
                      )}
                      {(onQueueAdd || onQueueRemove) && (() => {
                        const isQueued = queuedSet.has(p.result_id);
                        const queueDisabled = !isQueued && !eligibility.queueEligible;
                        return (
                          <button
                            className="refresh-btn"
                            style={{ fontSize: 10, padding: '1px 6px' }}
                            disabled={queueDisabled}
                            onClick={() => {
                              if (isQueued) {
                                onQueueRemove && onQueueRemove(p.result_id);
                              } else {
                                if (!eligibility.queueEligible) {
                                  return;
                                }
                                onQueueAdd && onQueueAdd({
                                  resultId: p.result_id,
                                  fingerprint: p.graph_fingerprint,
                                  source: 'report',
                                  architectureFamily: null,
                                  intent: queueIntent,
                                  queueEligible: eligibility.queueEligible,
                                  investigationEligible: eligibility.investigationEligible,
                                  validationEligible: eligibility.validationEligible,
                                  queueReason: eligibility.queueReason,
                                });
                              }
                            }}
                            title={isQueued
                              ? 'Remove from progression queue'
                              : queueDisabled
                                ? reportQueueReasonLabel(eligibility.queueReason)
                                : queueIntent === 'validation'
                                  ? 'Add to validation queue'
                                  : 'Add to investigation queue'}
                            aria-label={`${isQueued ? 'Remove' : 'Add'} ${p.result_id} ${isQueued ? 'from' : 'to'} investigation queue`}
                          >
                            {isQueued
                              ? 'Queued'
                              : queueDisabled
                                ? 'Ineligible'
                                : queueIntent === 'validation'
                                  ? 'Queue Validate'
                                  : 'Queue Investigate'}
                          </button>
                        );
                      })()}
                      {!eligibility.investigationEligible && !eligibility.validationEligible && (
                        <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                          {reportQueueReasonLabel(eligibility.queueReason)}
                        </span>
                      )}
                    </div>
                  )}
                </td>
              </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function StatCard({ label, value, color }) {
  return (
    <div style={{
      padding: '12px 16px', background: 'var(--bg-tertiary)', borderRadius: 6,
      borderLeft: `3px solid ${color || 'var(--accent-blue)'}`,
    }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: color || 'var(--text-primary)' }}>{value}</div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>{label}</div>
    </div>
  );
}

function EfficiencyChart({ frontier }) {
  if (!frontier || frontier.length === 0) return <p style={{ color: 'var(--text-muted)' }}>No Pareto-optimal programs yet.</p>;

  const W = 500, H = 200;
  const pad = { l: 60, r: 20, t: 20, b: 35 };

  const losses = frontier.map(p => p.final_loss || p.loss_ratio || 0).filter(l => isFinite(l));
  const flops = frontier.map(p => p.flops_forward || p.param_count || 0).filter(f => f > 0);
  if (losses.length < 2 || flops.length < 2) return null;

  const minL = Math.min(...losses), maxL = Math.max(...losses);
  const minF = Math.min(...flops), maxF = Math.max(...flops);
  const rangeL = maxL - minL || 1, rangeF = maxF - minF || 1;

  const xScale = v => pad.l + ((v - minF) / rangeF) * (W - pad.l - pad.r);
  const yScale = v => H - pad.b - ((v - minL) / rangeL) * (H - pad.t - pad.b);

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
      <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke="var(--border)" />
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} stroke="var(--border)" />
      <text x={W / 2} y={H - 5} textAnchor="middle" fill="var(--text-muted)" fontSize={10}>FLOPs / Params</text>
      <text x={12} y={H / 2} textAnchor="middle" fill="var(--text-muted)" fontSize={10} transform={`rotate(-90, 12, ${H / 2})`}>Loss</text>
      {frontier.map((p, i) => {
        const x = xScale(p.flops_forward || p.param_count || 0);
        const y = yScale(p.final_loss || p.loss_ratio || 0);
        if (!isFinite(x) || !isFinite(y)) return null;
        return (
          <circle key={i} cx={x} cy={y} r={5}
            fill="var(--accent-purple)" opacity={0.7}
            stroke="var(--bg-secondary)" strokeWidth={1.5}>
            <title>{p.graph_fingerprint?.slice(0, 10)}: loss={p.final_loss || p.loss_ratio}</title>
          </circle>
        );
      })}
    </svg>
  );
}

function generateMarkdown(data) {
  const s = data.summary || {};
  const s1Survivors = s.stage1_survivors ?? s.total_s1_passed ?? 0;
  const lines = [];
  lines.push('# Research Report');
  lines.push(`*Generated: ${new Date().toISOString()}*\n`);

  if (data.narrative) {
    lines.push('## Executive Summary\n');
    lines.push(data.narrative + '\n');
  }

  lines.push('## Key Statistics\n');
  lines.push(`- Total experiments: ${s.total_experiments || 0}`);
  lines.push(`- Programs evaluated: ${s.total_programs_evaluated || 0}`);
  lines.push(`- Stage 1 survivors: ${s1Survivors}`);
  lines.push(`- Novel discoveries: ${s.total_novel || 0}`);
  lines.push('');

  const top = data.top_programs || [];
  if (top.length > 0) {
    lines.push('## Discovery Rankings\n');
    lines.push('| Rank | Fingerprint | Loss Ratio | Novelty | Baseline | Similar To |');
    lines.push('|------|-------------|------------|---------|----------|------------|');
    top.forEach((p, i) => {
      lines.push(
        `| ${i + 1} | \`${(p.graph_fingerprint || '').slice(0, 12)}\` ` +
        `| ${p.loss_ratio != null ? p.loss_ratio.toFixed(4) : '--'} ` +
        `| ${p.novelty_score != null ? p.novelty_score.toFixed(3) : '--'} ` +
        `| ${p.baseline_loss_ratio != null ? p.baseline_loss_ratio.toFixed(3) : '--'} ` +
        `| ${p.most_similar_to || '--'} |`
      );
    });
    lines.push('');
  }

  const ops = data.op_success_rates || [];
  if (ops.length > 0) {
    lines.push('## Op Success Rates\n');
    lines.push('| Op | S1 Rate | Count |');
    lines.push('|----|---------|-------|');
    (Array.isArray(ops) ? ops : []).slice(0, 20).forEach(op => {
      lines.push(`| ${op.op_name || '?'} | ${op.s1_rate != null ? (op.s1_rate * 100).toFixed(1) + '%' : '--'} | ${op.total_count || '--'} |`);
    });
    lines.push('');
  }

  const failures = data.failure_patterns || {};
  if (Object.keys(failures).length > 0) {
    lines.push('## Failure Patterns\n');
    lines.push('```json');
    lines.push(JSON.stringify(failures, null, 2));
    lines.push('```\n');
  }

  const insights = data.insights || [];
  if (insights.length > 0) {
    lines.push('## Insights\n');
    insights.forEach(ins => {
      lines.push(`- **[${ins.category || 'general'}]** ${ins.content || ins}`);
    });
    lines.push('');
  }

  return lines.join('\n');
}

function NegativeResultsSummary() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/analytics/negative-results`)
      .then(r => r.ok ? r.json() : null)
      .then(d => setData(d))
      .catch(() => {});
  }, []);

  if (!data) return null;
  const hasContent = (data.failed_ops?.length > 0) || (data.refuted_hypotheses?.length > 0);
  if (!hasContent) return null;

  return (
    <div className="card">
      <div className="card-title">Do Not Pursue</div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Aggregated negative results: operations, patterns, and hypotheses that have been repeatedly tested and failed.
        Use this as a blacklist when designing future experiments.
      </p>
      <p style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 12, lineHeight: 1.5 }}>
        {data.summary}
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {data.failed_ops?.length > 0 && (
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>
              Zero-Success Ops
            </div>
            {data.failed_ops.slice(0, 8).map(op => (
              <div key={op.op_name} style={{
                display: 'flex', justifyContent: 'space-between', padding: '3px 0',
                borderBottom: '1px solid var(--border)',
              }}>
                <span style={{ fontSize: 12, fontFamily: 'monospace', color: 'var(--accent-red)' }}>{op.op_name}</span>
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>0/{op.n_used} S1</span>
              </div>
            ))}
          </div>
        )}
        {data.refuted_hypotheses?.length > 0 && (
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>
              Refuted Hypotheses
            </div>
            {data.refuted_hypotheses.slice(0, 5).map((h, i) => (
              <div key={i} style={{
                padding: '4px 6px', borderLeft: '2px solid var(--accent-red)',
                marginBottom: 4, fontSize: 11, color: 'var(--text-secondary)',
              }}>
                {(h.content || '').slice(0, 120)}{(h.content || '').length > 120 ? '...' : ''}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ResearchReport({ onSelectProgram, onSelectExperiment, onInvestigate, onValidate, onQueueAdd, onQueueRemove, queuedResultIds, eligibilityByResultId, onHypothesisHandoff }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/api/report`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(d => {
        setData(d);
        setLastUpdated(new Date());
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const handleExport = () => {
    if (!data) return;
    const md = generateMarkdown(data);
    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `research_report_${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (loading) return <div className="card"><p style={{ color: 'var(--text-muted)' }}>Loading report...</p></div>;
  if (error) return <div className="card"><p style={{ color: 'var(--accent-red)' }}>Error loading report: {error}</p></div>;
  if (!data) return null;

  const s = data.summary || {};
  const top = data.top_programs || [];
  const topExpanded = data.top_programs_expanded || [];
  const reportActionEligibility = data.action_eligibility || {};
  const mergedEligibilityByResultId = {
    ...(eligibilityByResultId || {}),
    ...reportActionEligibility,
  };
  const experiments = data.recent_experiments || [];
  const ops = data.op_success_rates || [];
  const failures = data.failure_patterns || {};
  const frontier = data.efficiency_frontier || [];
  const grammarWeights = data.grammar_weights || {};
  const insights = data.insights || [];
  const learningLog = data.learning_log || [];
  const crossRunStability = data.cross_run_stability || {};
  const mathFamilyCoverage = data.math_family_coverage || { families: [], totals: { n_tested: 0, n_survived: 0 } };
  const mathspaceOperatorImpact = data.mathspace_operator_impact || null;
  const routingModeComparison = data.routing_mode_comparison || null;
  const architectureRerunTelemetry = data.architecture_rerun_telemetry || {};
  const stabilitySummary = crossRunStability.summary || {};
  const stabilityCandidates = crossRunStability.candidates || [];

  const totalProg = s.total_programs_evaluated || 0;
  const s1Survivors = s.stage1_survivors ?? s.total_s1_passed ?? 0;
  const s1Rate = totalProg > 0 ? (s1Survivors / totalProg * 100).toFixed(1) : '0.0';

  // Separate best and worst ops
  const sortedOps = Array.isArray(ops)
    ? [...ops].sort((a, b) => (b.s1_rate || 0) - (a.s1_rate || 0))
    : [];
  const bestOps = sortedOps.filter(op => (op.s1_rate || 0) > 0).slice(0, 10);
  const worstOps = sortedOps.filter(op => (op.s1_rate || 0) === 0 && (op.total_count || 0) > 5).slice(0, 10);
  const confidenceFactors = {
    experiments: Math.min(1, (s.total_experiments || 0) / 5),
    programs: Math.min(1, totalProg / 500),
    rankings: Math.min(1, top.length / 10),
    opCoverage: Math.min(1, sortedOps.length / 8),
  };
  const confidenceScore = Math.round((
    confidenceFactors.experiments +
    confidenceFactors.programs +
    confidenceFactors.rankings +
    confidenceFactors.opCoverage
  ) / 4 * 100);
  const confidenceBand = confidenceScore >= 75
    ? { label: 'High confidence', color: 'var(--accent-green)' }
    : confidenceScore >= 45
      ? { label: 'Moderate confidence', color: 'var(--accent-yellow)' }
      : { label: 'Low confidence', color: 'var(--accent-red)' };
  const confidenceWarnings = [
    (s.total_experiments || 0) < 3 ? 'Fewer than 3 experiments: trends can change quickly with one additional run.' : null,
    totalProg < 200 ? `Only ${totalProg} programs evaluated: ranking order is still volatile.` : null,
    top.length < 5 ? 'Discovery ranking depth is shallow (<5 candidates).' : null,
    sortedOps.length < 4 ? 'Limited op-level coverage: “What Works” and “What Doesn’t Work” are early signals only.' : null,
  ].filter(Boolean);
  const confidenceStrengths = [
    (s.total_experiments || 0) >= 5 ? `${s.total_experiments || 0} experiments provide multi-run evidence.` : null,
    totalProg >= 500 ? `${totalProg.toLocaleString()} programs reduce random ranking swings.` : null,
    top.length >= 10 ? `${top.length} ranked discoveries improve selection confidence.` : null,
    sortedOps.length >= 8 ? `${sortedOps.length} ops observed gives broader operation-level signal.` : null,
  ].filter(Boolean);
  const decisionReadyCount = top.filter(program => decisionGate(program).decisionReady).length;
  const baselineEvidenceCount = top.filter(program => program.baseline_loss_ratio != null).length;
  const baselineWinCount = top.filter(program => program.baseline_loss_ratio != null && program.baseline_loss_ratio < 1.0).length;
  const baselineWinInterval = wilsonInterval(baselineWinCount, baselineEvidenceCount);
  const promotionEvidenceRows = top.map(program => promotionEvidence(program));
  const averagePromotionScore = promotionEvidenceRows.length > 0
    ? Math.round(promotionEvidenceRows.reduce((sum, row) => sum + row.score, 0) / promotionEvidenceRows.length)
    : 0;
  const reproducibilityRows = top.map(program => reproducibilityPacketStatus(program));
  const fullReproPacketCount = reproducibilityRows.filter(row => row.readyCount === row.totalChecks).length;
  const avgReproCompleteness = reproducibilityRows.length > 0
    ? Math.round((reproducibilityRows.reduce((sum, row) => sum + row.readyCount / row.totalChecks, 0) / reproducibilityRows.length) * 100)
    : 0;
  const uniqueFingerprintCount = Number(architectureRerunTelemetry.unique_fingerprint_count || 0);
  const totalResultRows = Number(architectureRerunTelemetry.total_result_rows || 0);
  const repeatResultRows = Number(architectureRerunTelemetry.repeat_result_rows || 0);
  const rerunRatioPercent = Number(architectureRerunTelemetry.rerun_ratio || 0) * 100;
  const topFingerprintConcentrationPercent = Number(architectureRerunTelemetry.top_fingerprint_concentration || 0) * 100;

  // Failure breakdown
  const failureByType = failures.by_error_type || failures;
  const failureByStage = failures.by_stage || {};

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Header + Export */}
      <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div className="card-title" style={{ marginBottom: 4 }}>Research Report</div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            Consolidated findings from {s.total_experiments || 0} experiments
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
            Last updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : 'loading'} · Source: /api/report
          </div>
        </div>
        <button className="start-btn" onClick={handleExport} style={{ padding: '8px 16px', fontSize: 13 }}>
          Export Markdown
        </button>
      </div>

      {/* Executive Summary */}
      <div className="card">
        <div className="card-title">Executive Summary</div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
          Fast snapshot of search productivity and quality. Use this first to decide whether to inspect rankings,
          failure patterns, or grammar updates in more detail.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 12, marginBottom: 16 }}>
          <StatCard label="Experiments" value={s.total_experiments || 0} color="var(--accent-blue)" />
          <StatCard label="Programs Tested" value={totalProg.toLocaleString()} color="var(--accent-purple)" />
          <StatCard label="S1 Survivors" value={s1Survivors} color="var(--accent-green)" />
          <StatCard label="S1 Pass Rate" value={`${s1Rate}%`} color={parseFloat(s1Rate) > 5 ? 'var(--accent-green)' : 'var(--accent-yellow)'} />
          <StatCard label="Novel" value={s.total_novel || 0} color="var(--accent-yellow)" />
        </div>
        {data.narrative && (
          <div style={{
            padding: 16, background: 'var(--bg-tertiary)', borderRadius: 6,
            borderLeft: '3px solid var(--accent-purple)', fontSize: 13,
            lineHeight: 1.6, color: 'var(--text-secondary)', whiteSpace: 'pre-wrap',
          }}>
            <div style={{ fontSize: 11, color: 'var(--accent-purple)', fontWeight: 600, marginBottom: 8, textTransform: 'uppercase' }}>
              Aria's Narrative
            </div>
            {data.narrative}
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-title">How to Read This Report</div>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          <div><strong>1. Discovery Rankings:</strong> pick candidates worth follow-up and open their full program details.</div>
          <div><strong>2. Timeline + What Works/Doesn't:</strong> verify whether trends are stable across experiments.</div>
          <div><strong>3. Grammar Evolution + Frontier:</strong> check if learned generation policy is moving toward better efficiency.</div>
          <div><strong>4. Insights:</strong> turn repeated patterns into next experiment hypotheses.</div>
        </div>
      </div>

      <div className="card">
        <div className="card-title">Metric Glossary</div>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          <div><strong>Loss Ratio:</strong> lower is better; compares post-training loss scale between candidates.</div>
          <div><strong>Baseline Loss Ratio:</strong> candidate loss versus fixed baseline; below 1.0 means candidate beats baseline.</div>
          <div><strong>Novelty Score:</strong> structural/behavioral difference signal; higher means less similar to prior programs.</div>
          <div><strong>Discovery Score:</strong> triage composite from loss, novelty, baseline comparison, and identity bonus.</div>
          <div><strong>S1 Survivor:</strong> program passed stage-1 learning evaluation and is eligible for deeper review.</div>
          <div><strong>CKA Source:</strong> `artifact` means reference-backed similarity; `fallback` means heuristic fallback path.</div>
        </div>
      </div>

      <div className="card">
        <div className="card-title">Confidence & Data Sufficiency</div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
          This callout estimates how stable current conclusions are based on sample size and coverage.
        </p>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
          <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Current confidence:</span>
          <span style={{ fontSize: 13, fontWeight: 700, color: confidenceBand.color }}>
            {confidenceBand.label} ({confidenceScore}%)
          </span>
        </div>
        <div style={{ display: 'grid', gap: 6, marginBottom: 10 }}>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Experiment depth: {(confidenceFactors.experiments * 100).toFixed(0)}%</div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Program volume: {(confidenceFactors.programs * 100).toFixed(0)}%</div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Ranking coverage: {(confidenceFactors.rankings * 100).toFixed(0)}%</div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Op coverage: {(confidenceFactors.opCoverage * 100).toFixed(0)}%</div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            Decision-ready candidates: {decisionReadyCount}/{top.length || 0}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            Avg promotion confidence (top set): {averagePromotionScore}%
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            Repro packet completeness: {avgReproCompleteness}% ({fullReproPacketCount}/{top.length || 0} fully ready)
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            Baseline win rate with evidence: {baselineEvidenceCount > 0 ? `${((baselineWinCount / baselineEvidenceCount) * 100).toFixed(1)}%` : 'n/a'}
            {baselineWinInterval ? ` (95% CI ${(baselineWinInterval.low * 100).toFixed(1)}-${(baselineWinInterval.high * 100).toFixed(1)}%)` : ''}
          </div>
        </div>
        {confidenceWarnings.length > 0 && (
          <div style={{ marginBottom: confidenceStrengths.length > 0 ? 8 : 0 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
              Cautions
            </div>
            <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              {confidenceWarnings.map((item, idx) => (
                <li key={`${item}-${idx}`}>{item}</li>
              ))}
            </ul>
          </div>
        )}
        {confidenceStrengths.length > 0 && (
          <div>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
              Supporting signals
            </div>
            <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              {confidenceStrengths.map((item, idx) => (
                <li key={`${item}-${idx}`}>{item}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-title">Unique Architectures vs Reruns</div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
          Concentration telemetry clarifies whether current learning signals come from architecture breadth
          or repeated reruns of a few fingerprints.
        </p>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 12, color: 'var(--text-secondary)' }}>
          <span><strong style={{ color: 'var(--accent-green)' }}>Unique fingerprints:</strong> {uniqueFingerprintCount}</span>
          <span><strong style={{ color: 'var(--text-muted)' }}>Rows:</strong> {totalResultRows}</span>
          <span><strong style={{ color: rerunRatioPercent >= 60 ? 'var(--accent-yellow)' : 'var(--text-muted)' }}>Rerun ratio:</strong> {rerunRatioPercent.toFixed(1)}%</span>
          <span><strong style={{ color: topFingerprintConcentrationPercent >= 35 ? 'var(--accent-yellow)' : 'var(--text-muted)' }}>Top fingerprint concentration:</strong> {topFingerprintConcentrationPercent.toFixed(1)}%</span>
        </div>
        <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-muted)' }}>
          Repeat rows: {repeatResultRows} · Weighting mode: {architectureRerunTelemetry.weighting_mode || 'unknown'}
        </div>
      </div>

      {/* Discovery Rankings */}
      {top.length > 0 && (
        <DiscoveryRankings
          programs={top}
          expandedPrograms={topExpanded}
          onSelectProgram={onSelectProgram}
          onInvestigate={onInvestigate}
          onValidate={onValidate}
          onQueueAdd={onQueueAdd}
          onQueueRemove={onQueueRemove}
          queuedResultIds={queuedResultIds}
          eligibilityByResultId={mergedEligibilityByResultId}
        />
      )}

      {/* Alternatives to Attention */}
      {top.length > 0 && <AlternativesToAttention programs={top} />}

      {/* Functional family coverage evidence */}
      {mathFamilyCoverage.families?.length > 0 && <FunctionalFamilyEvidence coverage={mathFamilyCoverage} />}

      {/* Mathspace operator impact evidence */}
      {mathspaceOperatorImpact?.available && <MathspaceOperatorImpact impact={mathspaceOperatorImpact} />}

      {/* Routing Mode Comparison */}
      {(top.length > 0 || routingModeComparison?.available) && (
        <RoutingModeComparison programs={top} comparison={routingModeComparison} />
      )}

      {/* Compression Technique Coverage */}
      {top.length > 0 && <CompressionTechniqueCoverage programs={top} />}

      {stabilityCandidates.length > 0 && (
        <div className="card">
          <div className="card-title">Cross-Run Stability</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
            Rank movement for top candidates across recent completed experiments. Use this to avoid overreacting to single-run spikes.
          </p>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 10, fontSize: 11, color: 'var(--text-muted)' }}>
            <span>Stable: {stabilitySummary.stable || 0}</span>
            <span>Up: {stabilitySummary.up || 0}</span>
            <span>Down: {stabilitySummary.down || 0}</span>
            <span>New: {stabilitySummary.new || 0}</span>
            <span>Window: {crossRunStability.window_size || 0} runs</span>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
                  <th scope="col" style={{ padding: '6px' }}>Fingerprint</th>
                  <th scope="col" style={{ padding: '6px' }}>Trend</th>
                  <th scope="col" style={{ padding: '6px' }}>Latest Rank</th>
                  <th scope="col" style={{ padding: '6px' }}>Previous Rank</th>
                  <th scope="col" style={{ padding: '6px' }}>Seen Runs</th>
                </tr>
              </thead>
              <tbody>
                {stabilityCandidates.slice(0, 12).map(candidate => {
                  const trendColor = candidate.trend === 'up'
                    ? 'var(--accent-green)'
                    : candidate.trend === 'down'
                      ? 'var(--accent-red)'
                      : candidate.trend === 'stable'
                        ? 'var(--accent-yellow)'
                        : 'var(--text-muted)';
                  return (
                    <tr key={candidate.result_id || candidate.graph_fingerprint} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td style={{ padding: '6px', fontFamily: 'monospace' }}>
                        {(candidate.graph_fingerprint || '').slice(0, 12)}
                      </td>
                      <td style={{ padding: '6px', color: trendColor, fontWeight: 600, textTransform: 'uppercase' }}>
                        {candidate.trend || 'unknown'}
                      </td>
                      <td style={{ padding: '6px' }}>{candidate.latest_rank ?? '--'}</td>
                      <td style={{ padding: '6px' }}>{candidate.previous_rank ?? '--'}</td>
                      <td style={{ padding: '6px' }}>{candidate.seen_runs ?? 0}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Experiment Timeline */}
      {experiments.length > 0 && (
        <div className="card">
          <div className="card-title">Experiment Timeline</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
            Chronological view of experiments showing how pass rates and discovery quality evolved over the search.
          </p>
          <div style={{ maxHeight: 400, overflowY: 'auto' }}>
            {experiments.map((exp, i) => {
              const s1 = exp.n_stage1_passed || 0;
              const total = exp.n_programs || 0;
              const confirmed = s1 > 0;
              return (
                <div key={exp.experiment_id || i} style={{
                  padding: '8px 12px', borderBottom: '1px solid var(--border)',
                  display: 'flex', gap: 12, alignItems: 'center',
                }}>
                  <span style={{
                    width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                    background: confirmed ? 'var(--accent-green)' : 'var(--accent-red)',
                  }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 12, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {exp.hypothesis ? `"${exp.hypothesis.slice(0, 80)}"` : `Experiment ${exp.experiment_id?.slice(0, 8)}`}
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      {exp.experiment_type || 'synthesis'} | {total} programs | {s1} S1 | {exp.created_at?.slice(0, 16)}
                    </div>
                  </div>
                  <span style={{
                    fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 4,
                    background: confirmed ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)',
                    color: confirmed ? 'var(--accent-green)' : 'var(--accent-red)',
                  }}>
                    {confirmed ? 'Confirmed' : 'Refuted'}
                  </span>
                  {onSelectExperiment && exp.experiment_id && (
                    <button
                      className="refresh-btn"
                      style={{ fontSize: 11, padding: '4px 8px', marginLeft: 8 }}
                      onClick={() => onSelectExperiment(exp.experiment_id)}
                      aria-label={`Open experiment ${exp.experiment_id}`}
                    >
                      Open
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* What Works + What Doesn't Work */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div className="card">
          <div className="card-title">What Works</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
            Operation types and patterns that consistently appear in successful architectures that passed Stage 1 learning evaluation.
          </p>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
            Use this section as a whitelist for future hypotheses: prioritize ops/combinations with repeatable S1 success.
          </p>
          {bestOps.length > 0 ? (
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Top Performing Ops</div>
              {bestOps.map((op, i) => (
                <div key={op.op_name || i} style={{
                  display: 'flex', justifyContent: 'space-between', padding: '4px 0',
                  borderBottom: '1px solid var(--border)',
                }}>
                  <span style={{ fontSize: 12, fontFamily: 'monospace' }}>{op.op_name}</span>
                  <span style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                    <span style={{ fontSize: 12, color: 'var(--accent-green)', fontWeight: 600 }}>
                      {((op.s1_rate || 0) * 100).toFixed(1)}%
                    </span>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      ({op.s1_count ?? Math.round((op.s1_rate || 0) * (op.total_count || 0))}/{op.total_count || 0})
                    </span>
                    <span
                      style={{
                        fontSize: 10,
                        fontWeight: 600,
                        textTransform: 'uppercase',
                        color: reliabilityBand(op.total_count || 0).color,
                      }}
                      title="Reliability from sample size: high (>=30), medium (12-29), low (<12)."
                    >
                      {reliabilityBand(op.total_count || 0).label}
                    </span>
                  </span>
                </div>
              ))}
            </div>
          ) : <p style={{ color: 'var(--text-muted)', fontSize: 12 }}>Insufficient data</p>}

          {data.structural_correlations && Object.keys(data.structural_correlations).length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Structural Correlations</div>
              {Object.entries(data.structural_correlations)
                .filter(([, v]) => Math.abs(v) > 0.1)
                .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                .slice(0, 8)
                .map(([key, val]) => (
                  <div key={key} style={{
                    display: 'flex', justifyContent: 'space-between', padding: '3px 0',
                    borderBottom: '1px solid var(--border)',
                  }}>
                    <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{key}</span>
                    <span style={{
                      fontSize: 11, fontWeight: 600,
                      color: val > 0 ? 'var(--accent-green)' : 'var(--accent-red)',
                    }}>
                      {val > 0 ? '+' : ''}{val.toFixed(3)}
                    </span>
                  </div>
                ))}
            </div>
          )}

          {data.top_op_combinations && data.top_op_combinations.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Best Op Combinations</div>
              {data.top_op_combinations.slice(0, 5).map((combo, i) => (
                <div key={i} style={{ fontSize: 11, padding: '3px 0', borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
                  {combo.ops ? combo.ops.join(' + ') : JSON.stringify(combo)}
                  {combo.s1_rate != null && (
                    <span style={{ marginLeft: 8, color: 'var(--accent-green)', fontWeight: 600 }}>
                      {(combo.s1_rate * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="card">
          <div className="card-title">What Doesn't Work</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
            Operation types and patterns that consistently lead to failure — compilation errors, numerical instability, or inability to learn.
          </p>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
            Use this section as a blacklist: reduce or constrain these patterns in upcoming runs to save search budget.
          </p>
          {Object.keys(failureByType).length > 0 || Object.keys(failureByStage).length > 0 ? (
            <>
              {Object.keys(failureByStage).length > 0 && (
                <div style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Failures by Stage</div>
                  {Object.entries(failureByStage).map(([stage, count]) => (
                    <div key={stage} style={{
                      display: 'flex', justifyContent: 'space-between', padding: '3px 0',
                      borderBottom: '1px solid var(--border)',
                    }}>
                      <span style={{ fontSize: 12 }}>{stage}</span>
                      <span style={{ fontSize: 12, color: 'var(--accent-red)' }}>{count}</span>
                    </div>
                  ))}
                </div>
              )}
              {typeof failureByType === 'object' && !Array.isArray(failureByType) && Object.keys(failureByType).length > 0 && (
                <div style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Failures by Error Type</div>
                  {Object.entries(failureByType).slice(0, 10).map(([errType, count]) => (
                    <div key={errType} style={{
                      display: 'flex', justifyContent: 'space-between', padding: '3px 0',
                      borderBottom: '1px solid var(--border)', gap: 8,
                    }}>
                      <span style={{ fontSize: 11, color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{errType}</span>
                      <span style={{ fontSize: 11, color: 'var(--accent-red)', flexShrink: 0 }}>{typeof count === 'number' ? count : JSON.stringify(count)}</span>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : <p style={{ color: 'var(--text-muted)', fontSize: 12 }}>No failure data yet</p>}

          {worstOps.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Worst Performing Ops (0% S1)</div>
              {worstOps.map((op, i) => (
                <div key={op.op_name || i} style={{
                  display: 'flex', justifyContent: 'space-between', padding: '3px 0',
                  borderBottom: '1px solid var(--border)',
                }}>
                  <span style={{ fontSize: 12, fontFamily: 'monospace', color: 'var(--text-secondary)' }}>{op.op_name}</span>
                  <span style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      0/{op.total_count || 0} S1
                    </span>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      ({op.total_count || 0} uses)
                    </span>
                    <span
                      style={{
                        fontSize: 10,
                        fontWeight: 600,
                        textTransform: 'uppercase',
                        color: reliabilityBand(op.total_count || 0).color,
                      }}
                      title="Reliability from sample size: high (>=30), medium (12-29), low (<12)."
                    >
                      {reliabilityBand(op.total_count || 0).label}
                    </span>
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Do Not Pursue */}
      <NegativeResultsSummary />

      {/* Grammar Evolution */}
      {grammarWeights.learned && grammarWeights.default && (
        <div className="card">
          <div className="card-title">Grammar Evolution</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
            How the generation weights shifted over time. Rising bars mean the system generates more of that operation; falling bars mean it learned to avoid it.
          </p>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
            Treat large weight deltas as policy changes; verify they align with the "What Works" and "What Doesn't Work" evidence above.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Weight Changes</div>
              {Object.keys({ ...grammarWeights.default, ...grammarWeights.learned }).sort().map(cat => {
                const old_w = grammarWeights.default[cat] || 1.0;
                const new_w = grammarWeights.learned ? (grammarWeights.learned[cat] || old_w) : old_w;
                const changed = Math.abs(new_w - old_w) > 0.1;
                return (
                  <div key={cat} style={{
                    display: 'flex', justifyContent: 'space-between', padding: '3px 0',
                    borderBottom: '1px solid var(--border)',
                    opacity: changed ? 1 : 0.5,
                  }}>
                    <span style={{ fontSize: 12 }}>{cat}</span>
                    <span style={{ fontSize: 12 }}>
                      <span style={{ color: 'var(--text-muted)' }}>{old_w.toFixed(1)}</span>
                      {changed && (
                        <>
                          <span style={{ color: 'var(--text-muted)', margin: '0 4px' }}>&rarr;</span>
                          <span style={{
                            fontWeight: 600,
                            color: new_w > old_w ? 'var(--accent-green)' : 'var(--accent-red)',
                          }}>
                            {new_w.toFixed(1)}
                          </span>
                        </>
                      )}
                    </span>
                  </div>
                );
              })}
            </div>
            {learningLog.length > 0 && (
              <div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase' }}>Recent Weight Changes</div>
                <div style={{ maxHeight: 200, overflowY: 'auto' }}>
                  {learningLog.slice(0, 10).map((entry, i) => (
                    <div key={i} style={{ padding: '4px 0', borderBottom: '1px solid var(--border)', fontSize: 11 }}>
                      <div style={{ color: 'var(--text-secondary)' }}>{entry.description || entry.event_type}</div>
                      <div style={{ color: 'var(--text-muted)' }}>{entry.created_at?.slice(0, 16)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Efficiency Frontier */}
      {frontier.length > 0 && (
        <div className="card">
          <div className="card-title">Efficiency Frontier</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
            Trade-off between model size (parameters) and learning speed (loss ratio). Points on the frontier are the best architectures at each size — nothing else learns faster for the same parameter budget.
          </p>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
            Choose candidates on this curve when you need better learning with limited compute budget.
          </p>
          <EfficiencyChart frontier={frontier} />
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
            {frontier.length} Pareto-optimal programs (lower loss, fewer FLOPs = better)
          </div>
        </div>
      )}

      {/* Insights / Recommendations */}
      {insights.length > 0 && (
        <div className="card">
          <div className="card-title">Insights & Recommendations</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
            Key takeaways and suggested next steps synthesized from all experiments.
          </p>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
            Convert high-confidence items into explicit hypotheses so they can be validated in campaign and timeline views.
          </p>
          {insights.slice(0, 15).map((ins, i) => (
            <div key={i} style={{
              padding: '8px 12px', borderBottom: '1px solid var(--border)',
              display: 'flex', gap: 8, alignItems: 'flex-start',
            }}>
              <span style={{
                fontSize: 10, fontWeight: 600, padding: '2px 6px', borderRadius: 3,
                background: 'var(--bg-tertiary)', color: 'var(--text-muted)',
                textTransform: 'uppercase', flexShrink: 0,
              }}>
                {ins.category || 'insight'}
              </span>
              <span style={{ fontSize: 12, color: 'var(--text-secondary)', flex: 1 }}>
                {ins.content || (typeof ins === 'string' ? ins : JSON.stringify(ins))}
              </span>
              {onHypothesisHandoff && (ins.category === 'hypothesis' || ins.category === 'success_factor') && (
                <button
                  className="refresh-btn"
                  style={{ fontSize: 10, padding: '1px 6px', flexShrink: 0 }}
                  onClick={() => onHypothesisHandoff({
                    source: 'report-insight',
                    hypothesis: ins.content || (typeof ins === 'string' ? ins : ''),
                    objective: `Test insight: ${(ins.content || '').slice(0, 80)}`,
                    suggestedMode: 'single',
                  })}
                  aria-label="Use this insight as experiment hypothesis"
                >
                  Use as Hypothesis
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default ResearchReport;
