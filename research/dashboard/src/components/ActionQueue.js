import { apiCall } from "../services/apiService";
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useEventBus } from '../hooks/useEventBus';
import { computeStrategy } from './StrategyAdvisor';
import { useAriaData } from '../hooks/useAriaData';

const BORDER_COLORS = {
  breakthrough: 'var(--accent-green)',
  warning: 'var(--accent-yellow)',
  strategy: 'var(--accent-blue)',
  healer: 'var(--accent-purple)',
  diagnosis: 'var(--accent-orange)',
  autonomous: 'var(--accent-cyan, #00BCD4)',
};

const ICONS = {
  trophy: '\uD83C\uDFC6',
  lightbulb: '\uD83D\uDCA1',
  warning: '\u26A0\uFE0F',
  wrench: '\uD83D\uDD27',
  stethoscope: '\uD83E\uDE7A',
  robot: '\uD83E\uDD16',
};

const TRUST_LEVELS = [
  { value: 'full', label: 'Full Autopilot', desc: 'Aria acts on everything, notifies on promotions and pivots' },
  { value: 'supervised', label: 'Supervised', desc: 'Aria acts but flags important decisions for review' },
  { value: 'advisory', label: 'Advisory', desc: 'Aria recommends, you approve (current behavior)' },
];

const BEHAVIOR_LABELS = {
  auto: 'Auto',
  notify: 'Notify',
  ask: 'Ask',
};

const BEHAVIOR_COLORS = {
  auto: 'var(--accent-green)',
  notify: 'var(--accent-yellow)',
  ask: 'var(--accent-blue)',
};

// ── Trust Level Selector ───────────────────────────────────────────

function TrustLevelSelector({ config, onUpdate }) {
  const [updating, setUpdating] = useState(false);

  const handleChange = async (newLevel) => {
    setUpdating(true);
    try {
      const res = await apiCall(`/api/aria/autonomy`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trust_level: newLevel }),
      });
      if (res.ok) {
        const updated = await res.json();
        onUpdate?.(updated);
      }
    } catch { /* ignore */ }
    setUpdating(false);
  };

  const currentLevel = config?.trust_level || 'supervised';

  return (
    <div style={{
      display: 'flex', gap: 4, alignItems: 'center', flexWrap: 'wrap',
      padding: '6px 0', marginBottom: 8,
    }}>
      <span style={{ fontSize: 11, color: 'var(--text-muted)', marginRight: 4 }}>Trust:</span>
      {TRUST_LEVELS.map(level => (
        <button
          key={level.value}
          onClick={() => handleChange(level.value)}
          disabled={updating}
          title={level.desc}
          style={{
            padding: '3px 10px', fontSize: 11, borderRadius: 4, cursor: 'pointer',
            border: `1px solid ${currentLevel === level.value ? 'var(--accent-cyan, #00BCD4)' : 'var(--border)'}`,
            background: currentLevel === level.value ? 'rgba(0, 188, 212, 0.15)' : 'transparent',
            color: currentLevel === level.value ? 'var(--accent-cyan, #00BCD4)' : 'var(--text-secondary)',
            fontWeight: currentLevel === level.value ? 600 : 400,
          }}
        >
          {level.label}
        </button>
      ))}
    </div>
  );
}

// ── Undo Timer Badge ───────────────────────────────────────────────

function UndoTimer({ seconds }) {
  if (seconds <= 0) return null;
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return (
    <span style={{
      fontSize: 10, color: 'var(--accent-yellow)', marginLeft: 6,
      fontFamily: 'monospace',
    }}>
      Undo: {mins}:{secs.toString().padStart(2, '0')}
    </span>
  );
}

// ── Main Component ─────────────────────────────────────────────────

function ActionQueue({
  onStart,
  onStop,
  onStartAutonomous,
  onStopAutonomous,
  isRunning,
  autonomousMode,
  onNavigateTab,
  onSelectProgram,
  onStrategyChange,
  dashboardData,
}) {
  const [actions, setActions] = useState([]);
  const [autonomyConfig, setAutonomyConfig] = useState(null);
  const [autonomousActions, setAutonomousActions] = useState([]);
  const [expandedId, setExpandedId] = useState(null);
  const [dismissing, setDismissing] = useState(null);
  const [approving, setApproving] = useState(null);
  const [startingAutonomous, setStartingAutonomous] = useState(false);
  const [showActivity, setShowActivity] = useState(false);
  const eventBus = useEventBus();
  const subscribe = eventBus?.subscribe;
  const { leaderboardEntries, learningTrajectory, mathFamilyCoverage, slowPollTick } = useAriaData() || {};
  const fetchRef = useRef(0);

  // Fetch computed action queue
  const fetchActions = useCallback(async () => {
    const id = ++fetchRef.current;
    try {
      const res = await apiCall(`/api/actions`);
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      if (id === fetchRef.current) {
        setActions(Array.isArray(data) ? data : []);
      }
    } catch {
      if (id !== fetchRef.current) return;
      const strategy = computeStrategy(dashboardData, leaderboardEntries, mathFamilyCoverage);
      if (strategy) {
        setActions([{
          id: `strategy_${strategy.id}`,
          type: 'strategy',
          priority: 5,
          icon: 'lightbulb',
          title: strategy.title,
          summary: strategy.rationale,
          detail: { tierSummary: strategy.tierSummary },
          actions: strategy.action
            ? [{ label: 'Execute', action: 'start', payload: strategy.action }]
            : [{ label: 'View Details', action: 'navigate', payload: { tab: 'discoveries' } }],
          dismissable: false,
          source: 'client_fallback',
        }]);
      }
    }
  }, [dashboardData, leaderboardEntries, mathFamilyCoverage]);

  // Fetch autonomy config
  const fetchAutonomyConfig = useCallback(async () => {
    try {
      const res = await apiCall(`/api/aria/autonomy`);
      if (res.ok) {
        const config = await res.json();
        setAutonomyConfig(config);
      }
    } catch { /* ignore */ }
  }, []);

  // Fetch autonomous activity
  const fetchActivity = useCallback(async () => {
    try {
      const res = await apiCall(`/api/aria/activity?limit=10`);
      if (res.ok) {
        const data = await res.json();
        setAutonomousActions(Array.isArray(data) ? data : []);
      }
    } catch { /* ignore */ }
  }, []);

  // Report active strategy to parent
  useEffect(() => {
    if (!onStrategyChange) return;
    const strategyAction = actions.find(a => a.type === 'strategy' || a.type === 'breakthrough');
    if (strategyAction) {
      onStrategyChange({ id: strategyAction.id, title: strategyAction.title, action: strategyAction.actions?.[0]?.payload });
    } else {
      onStrategyChange(null);
    }
  }, [actions, onStrategyChange]);

  useEffect(() => {
    fetchActions();
    fetchAutonomyConfig();
  }, [fetchActions, fetchAutonomyConfig]);

  useEffect(() => {
    fetchActions();
    if (showActivity) fetchActivity();
  }, [fetchActions, fetchActivity, showActivity, slowPollTick]);

  useEffect(() => {
    if (typeof subscribe !== 'function') return undefined;
    return subscribe('experiment_completed', fetchActions);
  }, [subscribe, fetchActions]);

  useEffect(() => {
    if (showActivity) fetchActivity();
  }, [showActivity, fetchActivity]);

  const handleDismiss = useCallback(async (actionId) => {
    setDismissing(actionId);
    try {
      await apiCall(`/api/actions/${actionId}/dismiss`, { method: 'POST' });
      setActions(prev => prev.filter(a => a.id !== actionId));
    } catch { /* ignore */ }
    setDismissing(null);
  }, []);

  const handleApprove = useCallback(async (actionId) => {
    setApproving(actionId);
    try {
      const res = await apiCall(`/api/actions/${actionId}/approve`, { method: 'POST' });
      if (res.ok) {
        fetchActions();
        fetchActivity();
      }
    } catch { /* ignore */ }
    setApproving(null);
  }, [fetchActions, fetchActivity]);

  const handleUndo = useCallback(async (actionId) => {
    try {
      const res = await apiCall(`/api/actions/${actionId}/undo`, { method: 'POST' });
      if (res.ok) {
        fetchActions();
        fetchActivity();
      }
    } catch { /* ignore */ }
  }, [fetchActions, fetchActivity]);

  const handleActionClick = useCallback((actionDef) => {
    if (!actionDef) return;
    const { action, payload } = actionDef;
    if (action === 'navigate' && payload?.tab) {
      onNavigateTab?.(payload.tab);
      if (payload.result_id) {
        onSelectProgram?.(payload.result_id);
      }
    } else if (action === 'start' && payload) {
      onStart?.({
        mode: payload.mode || payload.suggestedMode || 'continuous',
        source: payload.source || 'mixed',
        ...(payload.configOverrides || {}),
      });
    } else if (action === 'config_fix' && payload) {
      onNavigateTab?.('command');
    }
  }, [onNavigateTab, onSelectProgram, onStart]);

  const displayed = actions.slice(0, 4);

  return (
    <div className="action-queue" style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 16 }}>
      {/* Trust level selector */}
      <TrustLevelSelector config={autonomyConfig} onUpdate={setAutonomyConfig} />

      {/* Action cards */}
      {displayed.length === 0 ? (
        <div className="card action-queue-empty" style={{ padding: '12px 16px' }}>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', textAlign: 'center' }}>
            No actions needed — pipeline is healthy
          </div>
        </div>
      ) : (
        displayed.map(item => {
          const isExpanded = expandedId === item.id;
          const isAutonomous = item.source === 'autonomy' || item.behavior;
          const borderColor = isAutonomous
            ? BORDER_COLORS.autonomous
            : (BORDER_COLORS[item.type] || 'var(--border)');
          const icon = ICONS[item.icon] || (isAutonomous ? ICONS.robot : '');

          return (
            <div
              key={item.id || item.action_id}
              className="card action-card"
              style={{ borderLeft: `3px solid ${borderColor}`, padding: '10px 12px', cursor: 'pointer' }}
              onClick={() => setExpandedId(isExpanded ? null : (item.id || item.action_id))}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>
                    <span style={{ marginRight: 6 }}>{icon}</span>
                    {item.title}
                    {item.behavior && (
                      <span style={{
                        marginLeft: 8, fontSize: 10, padding: '1px 5px', borderRadius: 3,
                        color: BEHAVIOR_COLORS[item.behavior] || 'var(--text-muted)',
                        background: `${BEHAVIOR_COLORS[item.behavior] || 'var(--text-muted)'}22`,
                      }}>
                        {BEHAVIOR_LABELS[item.behavior] || item.behavior}
                      </span>
                    )}
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2, lineHeight: 1.4 }}>
                    {item.summary}
                  </div>
                  {item.undoable && (
                    <UndoTimer seconds={item.undo_remaining_seconds || 0} />
                  )}
                </div>
                <div style={{ display: 'flex', gap: 4, flexShrink: 0, alignItems: 'center' }}>
                  {/* Approve button for ASK-type pending actions */}
                  {item.status === 'pending' && item.behavior === 'ask' && (
                    <button
                      className="refresh-btn"
                      style={{ fontSize: 11, padding: '3px 8px', whiteSpace: 'nowrap', borderColor: 'var(--accent-green)', color: 'var(--accent-green)' }}
                      onClick={(e) => { e.stopPropagation(); handleApprove(item.action_id || item.id); }}
                    >
                      Approve
                    </button>
                  )}
                  {/* Undo button for recently executed actions */}
                  {item.undoable && (
                    <button
                      className="refresh-btn"
                      style={{ fontSize: 11, padding: '3px 8px', whiteSpace: 'nowrap', borderColor: 'var(--accent-yellow)', color: 'var(--accent-yellow)' }}
                      onClick={(e) => { e.stopPropagation(); handleUndo(item.action_id || item.id); }}
                    >
                      Undo
                    </button>
                  )}
                  {/* Standard action buttons */}
                  {item.actions?.map((act, i) => (
                    <button
                      key={i}
                      className="refresh-btn"
                      style={{ fontSize: 11, padding: '3px 8px', whiteSpace: 'nowrap' }}
                      onClick={(e) => { e.stopPropagation(); handleActionClick(act); }}
                    >
                      {act.label}
                    </button>
                  ))}
                  {/* "Let Aria Decide" for pending actions */}
                  {item.status === 'pending' && item.behavior === 'ask' && (
                    <button
                      className="refresh-btn"
                      disabled={approving === (item.action_id || item.id)}
                      style={{
                        fontSize: 11, padding: '3px 8px', whiteSpace: 'nowrap',
                        borderColor: approving === (item.action_id || item.id) ? 'var(--text-muted)' : 'var(--accent-cyan, #00BCD4)',
                        color: approving === (item.action_id || item.id) ? 'var(--text-muted)' : 'var(--accent-cyan, #00BCD4)',
                        opacity: approving === (item.action_id || item.id) ? 0.7 : 1,
                      }}
                      onClick={(e) => { e.stopPropagation(); handleApprove(item.action_id || item.id); }}
                      title="Grant Aria autonomy for this decision and similar future ones"
                    >
                      {approving === (item.action_id || item.id) ? 'Approving...' : 'Let Aria Decide'}
                    </button>
                  )}
                  {(item.dismissable !== false) && (
                    <button
                      className="refresh-btn"
                      style={{ fontSize: 11, padding: '3px 6px', color: 'var(--text-muted)' }}
                      onClick={(e) => { e.stopPropagation(); handleDismiss(item.id || item.action_id); }}
                      disabled={dismissing === (item.id || item.action_id)}
                      title="Dismiss"
                    >
                      &times;
                    </button>
                  )}
                </div>
              </div>

              {isExpanded && item.detail && (
                <div style={{
                  marginTop: 8, paddingTop: 8, borderTop: '1px solid var(--border)',
                  fontSize: 11, color: 'var(--text-muted)', fontFamily: 'monospace',
                  whiteSpace: 'pre-wrap', maxHeight: 120, overflow: 'auto',
                }}>
                  {JSON.stringify(item.detail, null, 2)}
                </div>
              )}
            </div>
          );
        })
      )}

      {/* Controls row: autonomous mode + activity toggle */}
      <div style={{ display: 'flex', gap: 8, marginTop: 4, flexWrap: 'wrap' }}>
        {!isRunning && !autonomousMode && (
          <button
            className="refresh-btn"
            disabled={startingAutonomous}
            style={{
              fontSize: 12, padding: '6px 12px', flex: 1,
              borderColor: startingAutonomous ? 'var(--text-muted)' : 'var(--accent-purple)',
              color: startingAutonomous ? 'var(--text-muted)' : 'var(--accent-purple)',
              opacity: startingAutonomous ? 0.7 : 1,
            }}
            onClick={async () => {
              setStartingAutonomous(true);
              try { await onStartAutonomous?.(); } catch { /* ignore */ }
              setStartingAutonomous(false);
            }}
          >
            {startingAutonomous ? 'Starting...' : 'Let Aria Decide'}
          </button>
        )}
        {autonomousMode && (
          <button
            className="refresh-btn"
            style={{ fontSize: 12, padding: '6px 12px', flex: 1, borderColor: 'var(--accent-red)', color: 'var(--accent-red)' }}
            onClick={() => onStopAutonomous?.()}
          >
            Stop Autonomous
          </button>
        )}
        {isRunning && !autonomousMode && (
          <button
            className="refresh-btn"
            style={{ fontSize: 12, padding: '6px 12px', flex: 1, borderColor: 'var(--accent-red)', color: 'var(--accent-red)' }}
            onClick={() => onStop?.()}
          >
            Stop Experiment
          </button>
        )}
        <button
          className="refresh-btn"
          style={{
            fontSize: 11, padding: '5px 10px',
            borderColor: showActivity ? 'var(--accent-cyan, #00BCD4)' : 'var(--border)',
            color: showActivity ? 'var(--accent-cyan, #00BCD4)' : 'var(--text-muted)',
          }}
          onClick={() => setShowActivity(s => !s)}
        >
          {showActivity ? 'Hide' : 'Show'} Activity
        </button>
      </div>

      {/* Autonomous activity feed */}
      {showActivity && autonomousActions.length > 0 && (
        <div style={{
          marginTop: 8, padding: '8px 12px',
          background: 'var(--bg-secondary)', borderRadius: 6,
          border: '1px solid var(--border)',
        }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase' }}>
            Recent Autonomous Activity
          </div>
          {autonomousActions.slice(0, 8).map((action, i) => (
            <div key={action.action_id || i} style={{
              fontSize: 11, padding: '4px 0',
              borderBottom: i < autonomousActions.length - 1 ? '1px solid var(--border)' : 'none',
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <span style={{
                  fontSize: 9, padding: '1px 4px', borderRadius: 2, marginRight: 6,
                  color: BEHAVIOR_COLORS[action.behavior] || 'var(--text-muted)',
                  background: `${BEHAVIOR_COLORS[action.behavior] || 'var(--text-muted)'}22`,
                  textTransform: 'uppercase', fontWeight: 600,
                }}>
                  {action.behavior || 'auto'}
                </span>
                <span style={{ color: 'var(--text-secondary)' }}>{action.title}</span>
                <span style={{
                  marginLeft: 6, fontSize: 10, padding: '1px 4px', borderRadius: 2,
                  color: action.status === 'executed' ? 'var(--accent-green)'
                    : action.status === 'undone' ? 'var(--accent-yellow)'
                    : action.status === 'failed' ? 'var(--accent-red)'
                    : 'var(--text-muted)',
                  background: action.status === 'executed' ? 'rgba(63, 185, 80, 0.12)'
                    : action.status === 'undone' ? 'rgba(210, 153, 34, 0.12)'
                    : action.status === 'failed' ? 'rgba(248, 81, 73, 0.12)'
                    : 'transparent',
                }}>
                  {action.status}
                </span>
              </div>
              <div style={{ flexShrink: 0, display: 'flex', alignItems: 'center', gap: 4 }}>
                {action.undoable && (
                  <button
                    className="refresh-btn"
                    style={{ fontSize: 10, padding: '1px 6px', borderColor: 'var(--accent-yellow)', color: 'var(--accent-yellow)' }}
                    onClick={() => handleUndo(action.action_id)}
                  >
                    Undo
                  </button>
                )}
                {action.created_at && (
                  <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                    {new Date(action.created_at * 1000).toLocaleTimeString()}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default ActionQueue;
