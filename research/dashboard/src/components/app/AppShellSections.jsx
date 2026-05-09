import React, { memo } from 'react';
import AriaAvatar from '../AriaAvatar';
import { NAV_CATEGORIES, TAB_LABELS, TAB_TIPS } from './appConfig';

const LAST_UPDATED_TIME_FORMATTER = new Intl.DateTimeFormat(undefined, {
  hour: 'numeric',
  minute: '2-digit',
  second: '2-digit',
});

function renderTabDelta(delta) {
  if (!delta) {
    return null;
  }
  return (
    <span style={{
      marginLeft: 4, fontSize: 9, fontWeight: 600, padding: '1px 4px',
      borderRadius: 3,
      background: delta.positive ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)',
      color: delta.positive ? 'var(--accent-green)' : 'var(--accent-red)',
      whiteSpace: 'nowrap',
    }}>
      {delta.text}
    </span>
  );
}

export const DashboardHeader = memo(function DashboardHeader({
  ariaMood,
  autoRefresh,
  dashboardUpdatedAt,
  fetchDashboard,
  isRunning,
  onAutoRefreshChange,
  onOpenDesigner,
  onToggleChat,
  onToggleHelp,
  onToggleSettings,
  showChat,
  showHelp,
  showSettings,
}) {
  const lastUpdatedLabel = dashboardUpdatedAt
    ? `Updated ${LAST_UPDATED_TIME_FORMATTER.format(new Date(dashboardUpdatedAt))}`
    : 'Loading...';

  return (
    <header className="app-header">
      <div className="header-left">
        <AriaAvatar mood={ariaMood} size={40} />
        <div>
          <h1>Dr. Aria Nexus</h1>
          <p className="subtitle">AI Research Scientist — Computational Architecture Discovery</p>
        </div>
        {isRunning && (
          <span className="header-running-badge">
            <span className="pulse-dot"></span>
            Running
          </span>
        )}
      </div>
      <div className="header-right">
        <div className="header-meta" aria-hidden="true">
          <span className="kbd-chip">Keys 1-7 · ? · Esc</span>
          <span className="last-updated-chip">{lastUpdatedLabel}</span>
        </div>
        <button
          className="refresh-btn"
          style={{ fontSize: 14, padding: '3px 8px', fontWeight: 700, lineHeight: 1, minWidth: 28 }}
          onClick={onToggleChat}
          aria-label="Toggle chat"
          aria-pressed={showChat}
          title="Aria Chat"
        >
          &#x1F4AC;
        </button>
        <button
          className="refresh-btn"
          style={{ fontSize: 14, padding: '3px 8px', fontWeight: 700, lineHeight: 1, minWidth: 28 }}
          onClick={onToggleSettings}
          aria-label="Toggle settings"
          aria-pressed={showSettings}
          title="Settings"
        >
          &#x2699;
        </button>
        <button
          className="refresh-btn"
          style={{ fontSize: 14, padding: '3px 8px', fontWeight: 700, lineHeight: 1, minWidth: 28 }}
          onClick={onToggleHelp}
          aria-label="Toggle help"
          aria-pressed={showHelp}
          title="Help (press ? key)"
        >
          ?
        </button>
        <label className="auto-refresh">
          <input
            type="checkbox"
            checked={autoRefresh}
            onChange={(e) => onAutoRefreshChange(e.target.checked)}
          />
          Auto-refresh
        </label>
        <button className="refresh-btn" onClick={fetchDashboard}>Refresh</button>
        <button
          className="refresh-btn"
          onClick={onOpenDesigner}
          title="Open Aria Designer with a blank canvas"
        >
          Designer
        </button>
      </div>
    </header>
  );
});

export const DashboardNav = memo(function DashboardNav({
  activeTab,
  primaryTab,
  setActiveTab,
  setSelectedExperiment,
  tabDeltas,
}) {
  return (
    <nav className="tab-nav-hierarchical">
      <div className="primary-nav">
        {Object.entries(NAV_CATEGORIES).map(([id, cat]) => (
          <button
            key={id}
            className={`primary-tab ${primaryTab === id ? 'active' : ''}`}
            onClick={() => {
              setActiveTab(cat.tabs[0]);
              setSelectedExperiment(null);
            }}
          >
            {cat.label}
          </button>
        ))}
      </div>
      <div className="secondary-nav">
        {NAV_CATEGORIES[primaryTab]?.tabs.map((tab) => (
          <button
            key={tab}
            className={`tab ${activeTab === tab ? 'active' : ''}`}
            title={TAB_TIPS[tab]}
            onClick={() => {
              setActiveTab(tab);
              setSelectedExperiment(null);
            }}
          >
            {TAB_LABELS[tab]}
            {renderTabDelta(tabDeltas[tab])}
          </button>
        ))}
      </div>
    </nav>
  );
});

export const DashboardStatusBanners = memo(function DashboardStatusBanners({
  actionError,
  actionNotice,
  activeTab,
  autonomousActive,
  blockedConfig,
  data,
  error,
  initialLoading,
  investigationQueue,
  onClearActionError,
  onClearActionNotice,
  onForceStart,
  onOpenLiveView,
  onQueueClear,
  onQueueCapabilityRank,
  onQueueConfirm,
  onQueueInvestigate,
  onQueueValidate,
  queueBreakdown,
  ariaCycle,
}) {
  return (
    <>
      {initialLoading && !error && activeTab !== 'reports' && (
        <div className="ux-state ux-state-loading" style={{ justifyContent: 'center', marginBottom: 14 }}>
          <span className="ux-spinner" />
          <div className="ux-stack">
            <span className="ux-state-title">Loading dashboard</span>
            <span className="ux-state-subtle">Fetching latest research, insights, and run status.</span>
          </div>
        </div>
      )}
      {error && (
        <div className="error-banner">
          Unable to connect to API: {error}
          <br />
          <small>Start the server: python -m research --mode=dashboard</small>
        </div>
      )}
      {actionError && (
        <div className="error-banner" style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }} onClick={onClearActionError}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span>{actionError}</span>
            {blockedConfig && (
              <button
                className="refresh-btn"
                style={{
                  fontSize: 11,
                  padding: '2px 8px',
                  borderColor: 'var(--accent-red)',
                  color: 'var(--accent-red)',
                  background: 'rgba(248, 81, 73, 0.1)',
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  onForceStart();
                }}
              >
                Force Start
              </button>
            )}
          </div>
          <button
            onClick={(e) => { e.stopPropagation(); onClearActionError(); }}
            aria-label="Dismiss error"
            style={{
              background: 'none', border: 'none', color: 'inherit', cursor: 'pointer',
              fontSize: 16, padding: '0 4px', opacity: 0.8, flexShrink: 0,
            }}
          >&times;</button>
        </div>
      )}
      {actionNotice?.message && (
        <div
          className="error-banner"
          style={{
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            borderColor: 'rgba(88, 166, 255, 0.45)',
            background: 'rgba(88, 166, 255, 0.10)',
            color: 'var(--accent-blue)',
          }}
          onClick={onClearActionNotice}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span>{actionNotice.message}</span>
          </div>
          <button
            onClick={(e) => { e.stopPropagation(); onClearActionNotice(); }}
            aria-label="Dismiss notice"
            style={{
              background: 'none', border: 'none', color: 'inherit', cursor: 'pointer',
              fontSize: 16, padding: '0 4px', opacity: 0.8, flexShrink: 0,
            }}
          >&times;</button>
        </div>
      )}
      {data?.is_running && (
        <div className="card" style={{ marginBottom: 12, padding: '10px 12px', borderLeft: '3px solid var(--accent-green)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              <strong style={{ color: 'var(--accent-green)' }}>
                {autonomousActive ? 'Autonomous active:' : 'Run active:'}
              </strong>{' '}
              {autonomousActive
                ? `${ariaCycle?.phase_label || ariaCycle?.phase || data?.progress?.status || 'running'}`
                : (data?.progress?.status || 'running')}
              {data?.progress?.experiment_id ? ` · ${String(data.progress.experiment_id).slice(0, 12)}` : ''}
              <span style={{ marginLeft: 8, color: 'var(--text-muted)' }}>
                {autonomousActive
                  ? 'Aria is iterating across cycles while you browse other tabs.'
                  : 'Search continues in background while you browse other tabs.'}
              </span>
            </div>
            {activeTab !== 'command' && (
              <button className="refresh-btn" onClick={onOpenLiveView}>
                Open live view
              </button>
            )}
          </div>
        </div>
      )}
      {investigationQueue.length > 0 && (
        <div className="card" style={{ marginBottom: 12, padding: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Progression Queue: {investigationQueue.length} candidate{investigationQueue.length === 1 ? '' : 's'} pinned
              {' '}({queueBreakdown.investigation} investigate, {queueBreakdown.capabilityRanking || 0} rank, {queueBreakdown.validation} validate, {queueBreakdown.confirmation || 0} confirm).
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button
                className="refresh-btn"
                onClick={onQueueInvestigate}
                disabled={queueBreakdown.investigation === 0}
              >
                Investigate Queue
              </button>
              <button
                className="refresh-btn"
                onClick={onQueueCapabilityRank}
                disabled={(queueBreakdown.capabilityRanking || 0) === 0}
              >
                Rank Queue
              </button>
              <button
                className="refresh-btn"
                onClick={onQueueValidate}
                disabled={queueBreakdown.validation === 0}
              >
                Validate Queue
              </button>
              <button
                className="refresh-btn"
                onClick={onQueueConfirm}
                disabled={(queueBreakdown.confirmation || 0) === 0}
              >
                Confirm Queue
              </button>
              <button className="refresh-btn" onClick={onQueueClear} style={{ marginLeft: 8, color: 'var(--accent-red)', borderColor: 'var(--accent-red)' }}>Clear Queue</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
});

export const DeferredInsightsSection = memo(function DeferredInsightsSection({
  children,
  emptyText,
  onLoad,
  title,
  visible,
}) {
  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>{title}</h3>
        {!visible && (
          <button
            className="refresh-btn"
            onClick={onLoad}
            style={{ fontSize: 12, padding: '4px 10px' }}
          >
            Load {title}
          </button>
        )}
      </div>
      {visible ? children : (
        <div className="card">
          <p style={{ color: 'var(--text-muted)', margin: 0 }}>
            {emptyText}
          </p>
        </div>
      )}
    </div>
  );
});
