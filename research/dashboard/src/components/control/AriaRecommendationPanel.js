import React from 'react';

export function AriaRecommendationPanel({ recommendation, onApply }) {
  if (!recommendation) return null;

  return (
    <div className="recommendation-section">
      <div className="recommendation-header">
        <strong>Aria's Recommendation</strong>
        {recommendation.confidence != null && (
          <span className="rec-confidence">
            Confidence: {(recommendation.confidence * 100).toFixed(0)}%
          </span>
        )}
      </div>
      <p className="recommendation-reasoning">{recommendation.reasoning}</p>
      {recommendation.config && Object.keys(recommendation.config).length > 0 && (
        <>
          <div className="recommendation-config">
            {Object.entries(recommendation.config).map(([k, v]) => (
              <span key={k} className="rec-param">{k}: {typeof v === 'object' && v !== null ? JSON.stringify(v) : String(v)}</span>
            ))}
          </div>
          <button className="apply-rec-btn" onClick={onApply}>
            Apply Suggestion
          </button>
        </>
      )}
    </div>
  );
}

export default AriaRecommendationPanel;
