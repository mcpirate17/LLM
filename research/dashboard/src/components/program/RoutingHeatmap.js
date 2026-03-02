import React from 'react';

export function RoutingHeatmap({ data, nExperts }) {
  if (!data || !Array.isArray(data)) return null;
  // data is [seq_len] or [seq_len, top_k]
  const seqLen = data.length;
  const experts = nExperts || 4;
  
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, marginBottom: 4 }}>
        Routing Heatmap (Token x Expert)
      </div>
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: `repeat(${experts}, 1fr)`,
        gap: 1,
        background: 'var(--border)',
        border: '1px solid var(--border)',
        padding: 1
      }}>
        {data.map((selected, t) => {
          const selectedList = Array.isArray(selected) ? selected : [selected];
          return Array.from({ length: experts }).map((_, e) => {
            const isActive = selectedList.includes(e);
            return (
              <div 
                key={`${t}-${e}`}
                title={`Token ${t}, Expert ${e}: ${isActive ? 'Active' : 'Inactive'}`}
                style={{
                  height: Math.max(2, 80 / seqLen),
                  background: isActive ? 'var(--accent-blue)' : 'var(--bg-secondary)',
                  opacity: isActive ? 0.8 : 1
                }}
              />
            );
          });
        })}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-muted)', marginTop: 2 }}>
        <span>T0 (start)</span>
        <span>T{seqLen - 1} (end)</span>
      </div>
    </div>
  );
}

export default RoutingHeatmap;
