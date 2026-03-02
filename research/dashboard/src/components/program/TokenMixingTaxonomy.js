import React from 'react';

const TOKEN_MIXING_OPS = {
  'attention': { family: 'attention', label: 'Attention', complexity: 'quadratic', desc: 'Standard QKV global mixing' },
  'linear_attention': { family: 'attention', label: 'Linear Attention', complexity: 'linear', desc: 'Kernelized global mixing' },
  'flash_attention': { family: 'attention', label: 'Flash Attention', complexity: 'quadratic', desc: 'IO-aware fast attention' },
  'mamba': { family: 'ssm', label: 'SSM (Mamba)', complexity: 'linear', desc: 'Selective state space' },
  'fourier_mixing': { family: 'spectral', label: 'Fourier', complexity: 'nlogn', desc: 'Global spectral mixing' },
  'shift_mixing': { family: 'spatial', label: 'Shift', complexity: 'linear', desc: 'Local spatial shifting' },
  'conv1d': { family: 'spatial', label: 'Conv1d', complexity: 'linear', desc: 'Local causal convolution' },
  'tropical_attention': { family: 'tropical', label: 'Tropical Attention', complexity: 'quadratic', desc: 'Max-plus semiring attention' },
};

const FAMILY_LABELS = { attention: 'Attn', ssm: 'SSM', spectral: 'Spectral', spatial: 'Spatial', tropical: 'Tropical' };
const FAMILY_COLORS = { attention: '#17a3ff', ssm: '#24d1a0', spectral: '#bc8cff', spatial: '#f0a030', tropical: '#ff5050' };

export function TokenMixingTaxonomy({ graphJson }) {
  if (!graphJson) return null;

  const rawNodes = graphJson.nodes || [];
  const nodes = Array.isArray(rawNodes) ? rawNodes : Object.values(rawNodes);
  const detected = [];
  const seen = new Set();
  for (const node of nodes) {
    if (!node || typeof node !== 'object') continue;
    const opName = node.op || node.op_name;
    if (opName && TOKEN_MIXING_OPS[opName] && !seen.has(opName)) {
      seen.add(opName);
      detected.push({ op: opName, ...TOKEN_MIXING_OPS[opName] });
    }
  }

  if (detected.length === 0) return null;

  const families = [...new Set(detected.map(d => d.family))];
  const hasQKV = families.includes('attention');
  const summary = hasQKV
    ? `Uses QKV-style attention${families.length > 1 ? ' + ' + families.filter(f => f !== 'attention').map(f => FAMILY_LABELS[f]).join(', ') : ''}`
    : `QKV-free: ${families.map(f => FAMILY_LABELS[f]).join(' + ')}`;

  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>
        Token Mixing Mechanism
      </div>
      <div style={{
        padding: '8px 12px', background: 'var(--bg-tertiary)', borderRadius: 6,
        borderLeft: `3px solid ${hasQKV ? 'var(--accent-blue)' : 'var(--accent-green)'}`,
      }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: hasQKV ? 'var(--accent-blue)' : 'var(--accent-green)', marginBottom: 6 }}>
          {summary}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {detected.map(d => (
            <div key={d.op} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11 }}>
              <span style={{
                padding: '1px 6px', borderRadius: 3, fontSize: 10, fontWeight: 600,
                background: `${FAMILY_COLORS[d.family]}22`, color: FAMILY_COLORS[d.family],
              }}>
                {FAMILY_LABELS[d.family]}
              </span>
              <code style={{ color: 'var(--text-secondary)', fontSize: 11 }}>{d.op}</code>
              <span style={{ color: 'var(--text-muted)' }}>{d.desc}</span>
              <span style={{
                marginLeft: 'auto', fontSize: 10, color: 'var(--text-muted)',
                fontStyle: 'italic', flexShrink: 0,
              }}>
                {d.complexity === 'linear' ? 'O(n)' : d.complexity === 'nlogn' ? 'O(n log n)' : 'O(n²)'}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default TokenMixingTaxonomy;
