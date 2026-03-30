import React from 'react';

export function StagePipeline({ program }) {
  const stages = [
    { key: 'stage0_passed', label: 'Stage 0', sublabel: 'Compilation' },
    { key: 'stage05_passed', label: 'Stage 0.5', sublabel: 'Stability' },
    { key: 'stage1_passed', label: 'Stage 1', sublabel: 'Learning' },
  ];

  return (
    <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
      {stages.map((stage, i) => {
        let passed = program[stage.key];
        // Infer: if a later stage passed, earlier stages must have too
        if (passed == null) {
          if (stage.key === 'stage05_passed' && program.stage1_passed) passed = 1;
          if (stage.key === 'stage0_passed' && (program.stage05_passed || program.stage1_passed)) passed = 1;
        }
        const color = passed ? 'var(--accent-green)' : 'var(--accent-red)';
        const bg = passed ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)';
        return (
          <React.Fragment key={stage.key}>
            {i > 0 && <span style={{ color: 'var(--text-muted)', fontSize: 16 }}>&rarr;</span>}
            <div style={{
              padding: '6px 12px',
              background: bg,
              border: `1px solid ${color}`,
              borderRadius: 6,
              textAlign: 'center',
              minWidth: 80,
            }}>
              <div style={{ fontSize: 12, fontWeight: 600, color }}>{passed ? 'PASS' : 'FAIL'}</div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{stage.label}</div>
            </div>
          </React.Fragment>
        );
      })}
    </div>
  );
}

export default React.memo(StagePipeline);
