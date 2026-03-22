import React from 'react';
import useProgramData from '../hooks/useProgramData';
import ProgramHeader from './programDetail/ProgramHeader';
import TrainingMetricsPanel, { CoreMetricsColumn } from './programDetail/TrainingMetricsPanel';
import ArchitectureView, { FingerprintColumn } from './programDetail/ArchitectureView';
import EvalResultsPanel from './programDetail/EvalResultsPanel';

function ProgramDetail({ resultId, onClose, onActionComplete, onSelectExperiment, onViewInLeaderboard, onSelectCampaign, onOpenInDesigner, onAddToComparison, eligibilityByResultId, defaultOverrideIneligible = false }) {
  const {
    state,
    dispatch,
    tier,
    alreadyInvestigated,
    alreadyValidated,
    canInvestigate,
    canValidate,
    handleLaunchRefinement,
    fetchDecisionPacket,
    fetchAndCopyManifest,
    manifestCopied,
    drawerResizeRef,
  } = useProgramData({ resultId, defaultOverrideIneligible, onActionComplete, onClose, eligibilityByResultId });

  const {
    program, loading, error, leaderboardEntry,
    linkedHypothesis, linkedDecision, linkedExperiment, linkedCampaign,
    latestRefineLaunch, refineLaunchHistory, refineTrace, refineTraceLoading,
    refineAnalysis, refineAnalysisLoading, refineAnalysisError,
    actionStarting, drawerWidthVw, drawerMaximized, resizingDrawer,
  } = state;

  if (!resultId) return null;

  const fmt = (v, d = 4) => v != null ? Number(v).toFixed(d) : '--';
  const fmtMs = v => v != null ? `${Number(v).toFixed(1)}ms` : '--';
  const fmtMem = v => v != null ? `${Number(v).toFixed(1)}MB` : '--';
  const fmtInt = v => v != null ? Number(v).toLocaleString() : '--';
  const shortId = (v, n = 12) => { const s = String(v || '').trim(); return !s ? '--' : (s.length > n ? s.slice(0, n) : s); };
  const formatUnixTimestamp = (value) => {
    if (value == null) return null;
    const n = Number(value);
    if (!Number.isFinite(n)) return null;
    const ms = n > 1e12 ? n : n * 1000;
    return new Date(ms).toLocaleString();
  };

  return (
    <div className="program-drawer-backdrop" onMouseDown={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div
        className="program-drawer"
        onClick={e => e.stopPropagation()}
        onMouseDown={e => e.stopPropagation()}
        style={drawerMaximized
          ? { width: '100%', minWidth: 0, maxWidth: '100%' }
          : { width: `${drawerWidthVw}vw`, minWidth: 460, maxWidth: '95vw' }}
      >
        {!drawerMaximized && (
          <div
            onMouseDown={(event) => {
              event.preventDefault();
              event.stopPropagation();
              drawerResizeRef.current = { startX: event.clientX, startVw: drawerWidthVw };
              dispatch({ type: 'SET_DRAWER', payload: { resizingDrawer: true } });
            }}
            style={{
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: 0,
              width: 8,
              cursor: 'col-resize',
              background: resizingDrawer ? 'rgba(88, 166, 255, 0.2)' : 'transparent',
              borderLeft: '1px solid rgba(88, 166, 255, 0.35)',
              zIndex: 2,
            }}
            title="Drag to resize"
            aria-hidden="true"
          />
        )}
        <div className="program-drawer-header">
          <span>Program Detail</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <button
              className="refresh-btn"
              aria-pressed={drawerMaximized}
              onClick={() => dispatch({ type: 'SET_DRAWER', payload: { drawerMaximized: !drawerMaximized } })}
              style={{ fontSize: 16, padding: '4px 8px' }}
              title={drawerMaximized ? 'Exit fullscreen' : 'Expand to fullscreen'}
            >
              {drawerMaximized ? '\u2750' : '\u2922'}
            </button>
            <button className="close-btn" onClick={onClose}>&times;</button>
          </div>
        </div>
        <div style={{ flex: 1, overflowY: 'auto', padding: 24 }}>
        {loading ? (
          <p style={{ color: 'var(--text-muted)' }}>Loading...</p>
        ) : error ? (
          <p style={{ color: 'var(--accent-red)' }}>{error}</p>
        ) : !program ? (
          <p style={{ color: 'var(--accent-red)' }}>Program not found</p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <ProgramHeader
              program={program}
              leaderboardEntry={leaderboardEntry}
              resultId={resultId}
              dispatch={dispatch}
              state={state}
              linkedHypothesis={linkedHypothesis}
              linkedDecision={linkedDecision}
              linkedExperiment={linkedExperiment}
              linkedCampaign={linkedCampaign}
              fetchDecisionPacket={fetchDecisionPacket}
              fetchAndCopyManifest={fetchAndCopyManifest}
              manifestCopied={manifestCopied}
              onClose={onClose}
              onSelectExperiment={onSelectExperiment}
              onAddToComparison={onAddToComparison}
              onOpenInDesigner={onOpenInDesigner}
              onViewInLeaderboard={onViewInLeaderboard}
              onSelectCampaign={onSelectCampaign}
              latestRefineLaunch={latestRefineLaunch}
              refineLaunchHistory={refineLaunchHistory}
              refineTrace={refineTrace}
              refineTraceLoading={refineTraceLoading}
              fmt={fmt}
              fmtInt={fmtInt}
              shortId={shortId}
              formatUnixTimestamp={formatUnixTimestamp}
            />
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <CoreMetricsColumn
                program={program}
                leaderboardEntry={leaderboardEntry}
                refineAnalysis={refineAnalysis}
                fmt={fmt}
                fmtMs={fmtMs}
                fmtMem={fmtMem}
                fmtInt={fmtInt}
              />
              <FingerprintColumn
                program={program}
                fmtMs={fmtMs}
                fmtMem={fmtMem}
                fmtInt={fmtInt}
              />
            </div>
            <TrainingMetricsPanel
              program={program}
              resultId={resultId}
              linkedHypothesis={linkedHypothesis}
              linkedDecision={linkedDecision}
              fmt={fmt}
              fmtMs={fmtMs}
            />
            <ArchitectureView
              program={program}
              leaderboardEntry={leaderboardEntry}
              resultId={resultId}
              refineAnalysis={refineAnalysis}
              refineAnalysisLoading={refineAnalysisLoading}
              refineAnalysisError={refineAnalysisError}
              handleLaunchRefinement={handleLaunchRefinement}
              actionStarting={actionStarting}
              onViewInLeaderboard={onViewInLeaderboard}
              onOpenInDesigner={onOpenInDesigner}
              onClose={onClose}
            />
            <EvalResultsPanel
              program={program}
              leaderboardEntry={leaderboardEntry}
              resultId={resultId}
              dispatch={dispatch}
              state={state}
              onActionComplete={onActionComplete}
              onClose={onClose}
              canInvestigate={canInvestigate}
              canValidate={canValidate}
              alreadyInvestigated={alreadyInvestigated}
              alreadyValidated={alreadyValidated}
            />
          </div>
        )}
        </div>
      </div>
    </div>
  );
}

export default ProgramDetail;
