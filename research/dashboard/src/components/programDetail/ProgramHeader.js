import React from 'react';
import { ProgramHeaderSection, ProvenancePanel, DecisionPacketPanel, RefinementTracePanel } from './InfoPanels';

function ProgramHeader({
  program, leaderboardEntry, resultId, dispatch, state,
  linkedHypothesis, linkedDecision, linkedExperiment, linkedCampaign,
  fetchDecisionPacket, fetchAndCopyManifest, manifestCopied,
  onClose, onSelectExperiment, onAddToComparison, onOpenInDesigner,
  onViewInLeaderboard, onSelectCampaign,
  latestRefineLaunch, refineLaunchHistory, refineTrace, refineTraceLoading,
  fmt, fmtInt, shortId, formatUnixTimestamp,
}) {
  const {
    provenanceOpen, decisionPacket, decisionPacketLoading,
    decisionPacketError, decisionPacketOpen, manifestLoading,
    actionError,
  } = state;

  const refinementTraceLabels = {
    panelTitle: 'Refinement Trace',
    openRefinementRun: 'Open Refinement Run',
    viewTopRefinedResult: 'View Top Refined Result',
    newFingerprints: 'New Fingerprints',
    openFingerprint: 'Open Fingerprint',
    recentRefinementLaunches: 'Recent Refinement Launches',
  };

  return (
    <>
      <ProgramHeaderSection program={program} leaderboardEntry={leaderboardEntry} />
      <ProvenancePanel
        program={program}
        linkedHypothesis={linkedHypothesis}
        leaderboardEntry={leaderboardEntry}
        linkedCampaign={linkedCampaign}
        linkedExperiment={linkedExperiment}
        provenanceOpen={provenanceOpen}
        setProvenanceOpen={(val) => dispatch({ type: 'SET_UI', payload: { provenanceOpen: val } })}
        formatUnixTimestamp={formatUnixTimestamp}
        onClose={onClose}
        onSelectExperiment={onSelectExperiment}
        onAddToComparison={onAddToComparison}
        onOpenInDesigner={onOpenInDesigner}
        onViewInLeaderboard={onViewInLeaderboard}
        onSelectCampaign={onSelectCampaign}
        decisionPacket={decisionPacket}
        decisionPacketLoading={decisionPacketLoading}
        fetchDecisionPacket={fetchDecisionPacket}
        manifestLoading={manifestLoading}
        manifestCopied={manifestCopied}
        fetchAndCopyManifest={fetchAndCopyManifest}
        resultId={resultId}
      />
      <DecisionPacketPanel
        decisionPacket={decisionPacket}
        decisionPacketError={decisionPacketError}
        decisionPacketOpen={decisionPacketOpen}
        setDecisionPacketOpen={(val) => dispatch({ type: 'SET_UI', payload: { decisionPacketOpen: val } })}
      />
      {/* Error displays */}
      {(program.error_message || program.stage0_error) && (
        <div style={{
          padding: 8,
          background: 'rgba(248, 81, 73, 0.1)',
          border: '1px solid var(--accent-red)',
          borderRadius: 4,
          fontSize: 12,
          fontFamily: 'monospace',
          color: 'var(--accent-red)',
        }}>
          {program.error_type && (
            <span style={{ fontWeight: 600 }}>[{program.error_type}] </span>
          )}
          {program.error_message || program.stage0_error}
        </div>
      )}
      {actionError && (
        <div style={{
          padding: 8,
          background: 'rgba(248, 81, 73, 0.1)',
          border: '1px solid var(--accent-red)',
          borderRadius: 4,
          fontSize: 12,
          color: 'var(--accent-red)',
        }}>
          {actionError}
        </div>
      )}
      {/* Refinement Trace */}
      <RefinementTracePanel
        latestRefineLaunch={latestRefineLaunch}
        labels={refinementTraceLabels}
        shortId={shortId}
        onClose={onClose}
        onSelectExperiment={onSelectExperiment}
        onViewInLeaderboard={onViewInLeaderboard}
        refineTraceLoading={refineTraceLoading}
        refineTrace={refineTrace}
        fmtInt={fmtInt}
        fmt={fmt}
        refineLaunchHistory={refineLaunchHistory}
      />
    </>
  );
}

export default React.memo(ProgramHeader);
