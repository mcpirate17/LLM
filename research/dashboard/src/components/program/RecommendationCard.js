import React from 'react';

export function RecommendationCard({ recommendation }) {
  if (!recommendation) return null;
  const { title, description, reasoning, confidence, actions = [] } = recommendation;
  
  return (
    <div style={{
      padding: 12,
      background: 'var(--bg-tertiary)',
      borderRadius: 8,
      border: '1px solid var(--border)',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>{title}</div>
        <div style={{ fontSize: 10, padding: '2px 6px', borderRadius: 4, background: 'rgba(0, 212, 255, 0.1)', color: 'var(--accent-purple)' }}>
          {Math.round(confidence * 100)}% Confidence
        </div>
      </div>
      
      <div style={{ fontSize: 12, color: 'var(--text-primary)', marginBottom: 8 }}>{description}</div>
      <div style={{ fontSize: 11, color: 'var(--text-secondary)', fontStyle: 'italic', marginBottom: 12 }}>{reasoning}</div>
      
      {actions.length > 0 && (
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {actions.map((action, i) => (
            <button key={i} style={{ 
              fontSize: 10, padding: '4px 8px', borderRadius: 4, 
              background: 'var(--bg-secondary)', border: '1px solid var(--border)', 
              color: 'var(--text-primary)', cursor: 'pointer' 
            }}>
              {action.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default React.memo(RecommendationCard);
