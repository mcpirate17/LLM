import React from 'react';

export default function ChartActionRail({ insight, recommendation, actions = [] }) {
  const visibleActions = actions.filter((action) => typeof action?.onClick === 'function');
  if (!insight && !recommendation && visibleActions.length === 0) return null;

  return (
    <div className="chart-action-rail">
      <div className="chart-action-copy">
        {insight ? <div className="chart-action-insight">{insight}</div> : null}
        {recommendation ? <div className="chart-action-recommendation">{recommendation}</div> : null}
      </div>
      {visibleActions.length > 0 ? (
        <div className="chart-action-buttons">
          {visibleActions.map((action) => (
            <button
              key={action.id || action.label}
              type="button"
              className={`btn-small ${action.variant === 'secondary' ? 'btn-secondary' : ''}`.trim()}
              onClick={action.onClick}
            >
              {action.label}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
