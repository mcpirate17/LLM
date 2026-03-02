import React from 'react';

export function AriaAdvice({ analysis }) {
  const advice = analysis?.brittleness_advice;
  if (!advice) return null;

  return (
    <div style={{
      marginTop: 12,
      padding: 12,
      background: 'rgba(0, 212, 255, 0.05)',
      borderRadius: 8,
      border: '1px solid rgba(0, 212, 255, 0.3)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <span style={{ fontSize: 16 }}>🧬</span>
        <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--accent-purple)', textTransform: 'uppercase' }}>
          Aria's Advice: Stabilisation
        </div>
      </div>
      
      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6, color: 'var(--text-primary)' }}>
        {advice.summary}
      </div>
      
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 10, lineHeight: 1.4 }}>
        {advice.diagnosis}
      </div>

      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 4 }}>
          Recommended Improvements
        </div>
        <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: 'var(--text-primary)', lineHeight: 1.5 }}>
          {advice.remedies.map((r, i) => (
            <li key={i} style={{ marginBottom: 2 }}>{r}</li>
          ))}
        </ul>
      </div>

      <div style={{ 
        padding: '8px 10px', 
        background: 'rgba(0, 212, 255, 0.1)', 
        borderRadius: 6,
        fontSize: 11,
        fontStyle: 'italic',
        color: 'var(--text-secondary)',
        lineHeight: 1.4,
        borderLeft: '2px solid var(--accent-purple)'
      }}>
        " {advice.aria_insight} "
      </div>
    </div>
  );
}

export default AriaAdvice;
