import { memo, useEffect, useState } from 'react';
import '../styles/HelpPanel.css';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8091';

function ContextualTip({ componentId }) {
  const [tips, setTips] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!componentId) {
      setTips(null);
      return;
    }
    const leafId = componentId.includes('/') ? componentId.split('/').pop() : componentId;
    if (!leafId) return;

    setLoading(true);
    const controller = new AbortController();
    fetch(`${API_BASE}/api/v1/help/component/${encodeURIComponent(leafId)}/tips`, {
      signal: controller.signal,
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => { setTips(data); setLoading(false); })
      .catch(() => { setLoading(false); });

    return () => controller.abort();
  }, [componentId]);

  if (!componentId || loading || !tips) return null;

  const worksWell = tips.works_well_with || [];
  const patterns = tips.patterns || [];
  const leaderboardUsage = tips.leaderboard_usage;

  if (!worksWell.length && !patterns.length) return null;

  return (
    <div className="contextual-tip">
      {leaderboardUsage && (
        <div className="tip-badge tip-badge-leaderboard">{leaderboardUsage}</div>
      )}
      {worksWell.length > 0 && (
        <div className="tip-row">
          <span className="tip-label">Works well with:</span>
          <span className="tip-chips">
            {worksWell.slice(0, 5).map((id) => (
              <span key={id} className="tip-chip tip-chip-good">{id}</span>
            ))}
          </span>
        </div>
      )}
      {tips.avoid_with?.length > 0 && (
        <div className="tip-row">
          <span className="tip-label">Avoid with:</span>
          <span className="tip-chips">
            {tips.avoid_with.slice(0, 3).map((id) => (
              <span key={id} className="tip-chip tip-chip-bad">{id}</span>
            ))}
          </span>
        </div>
      )}
      {patterns.length > 0 && (
        <div className="tip-pattern">{patterns[0]}</div>
      )}
      {tips.research_warnings?.length > 0 && (
        <div className="tip-warning">{tips.research_warnings[0]}</div>
      )}
    </div>
  );
}

export default memo(ContextualTip);
