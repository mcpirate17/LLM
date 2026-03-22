import React from 'react';
import StagePipeline from '../program/StagePipeline';
import TierBadge from '../shared/TierBadge';
import EvidenceFlagChips from '../program/EvidenceFlagChips';
import HypothesisLineage from '../program/HypothesisLineage';
import OutcomesByPhase from '../program/OutcomesByPhase';
import FailureContext from '../program/FailureContext';
import RecommendationCard from '../program/RecommendationCard';

export function ProgramHeaderSection({ program, leaderboardEntry }) {
  if (!program) return null;
  const headerId = program.graph_fingerprint || program.reference_name || program.result_id || 'unknown';
  return (
    <div>
      <div style={{ fontFamily: 'monospace', fontSize: 13, color: 'var(--accent-blue)', marginBottom: 4, display: 'flex', alignItems: 'center', gap: 8 }}>
        <span>{headerId}</span>
        {program.stage_at_death && program.stage_at_death !== 'survived' && (
          <span style={{ fontSize: 11, color: 'var(--accent-red)' }}>
            died at {program.stage_at_death}
          </span>
        )}
        {program.is_reference && (
          <span style={{ fontSize: 10, color: 'var(--accent-purple)', border: '1px solid var(--accent-purple)', borderRadius: 4, padding: '1px 6px' }}>
            REFERENCE
          </span>
        )}
        {leaderboardEntry && (
          <>
            <TierBadge tier={leaderboardEntry.tier} entry={leaderboardEntry} />
            <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--accent-green)' }}>
              Score: {Number(leaderboardEntry.composite_score).toFixed(3)}
            </span>
          </>
        )}
      </div>
      <StagePipeline program={program} />
    </div>
  );
}

export function ProvenancePanel({
  program,
  linkedHypothesis,
  leaderboardEntry,
  linkedCampaign,
  linkedExperiment,
  provenanceOpen,
  setProvenanceOpen,
  formatUnixTimestamp,
  onClose,
  onSelectExperiment,
  onAddToComparison,
  onOpenInDesigner,
  onViewInLeaderboard,
  onSelectCampaign,
  decisionPacket,
  decisionPacketLoading,
  fetchDecisionPacket,
  manifestLoading,
  manifestCopied,
  fetchAndCopyManifest,
  resultId,
}) {
  if (!(program?.experiment_id || linkedHypothesis || leaderboardEntry || linkedCampaign)) return null;
  return (
    <div style={{ background: 'var(--bg-tertiary)', borderRadius: 6, border: '1px solid var(--border)', overflow: 'hidden' }}>
      <div
        onClick={() => setProvenanceOpen(!provenanceOpen)}
        style={{ padding: '8px 12px', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center', userSelect: 'none' }}
      >
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
          Provenance & Context
        </span>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          {provenanceOpen ? '▾ collapse' : '▸ expand'}
        </span>
      </div>
      {provenanceOpen && (
        <div style={{ padding: '0 12px 12px', display: 'flex', flexDirection: 'column', gap: 8 }}>
          {program?.experiment_id && (
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <span style={{ color: 'var(--text-muted)', fontWeight: 600, minWidth: 90 }}>Experiment:</span>
              <span style={{ fontFamily: 'monospace', fontSize: 11 }}>{program.experiment_id.slice(0, 12)}</span>
              {linkedExperiment?.started_at && (
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  {formatUnixTimestamp(linkedExperiment.started_at)}
                </span>
              )}
              {linkedExperiment?.experiment_type && (
                <span style={{ fontSize: 10, color: 'var(--accent-blue)', border: '1px solid var(--accent-blue)', borderRadius: 3, padding: '0 4px' }}>
                  {linkedExperiment.experiment_type}
                </span>
              )}
            </div>
          )}
          {linkedHypothesis && (
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'flex', alignItems: 'baseline', gap: 8 }}>
              <span style={{ color: 'var(--text-muted)', fontWeight: 600, minWidth: 90, flexShrink: 0 }}>Hypothesis:</span>
              <span
                style={{
                  overflowWrap: 'anywhere',
                  color: linkedHypothesis.status === 'confirmed' ? 'var(--accent-green)'
                    : linkedHypothesis.status === 'refuted' ? 'var(--accent-red)' : 'var(--text-secondary)',
                }}
                title={linkedHypothesis.prediction}
              >
                [{linkedHypothesis.status}] {linkedHypothesis.prediction}
              </span>
            </div>
          )}
          {linkedCampaign && (
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ color: 'var(--text-muted)', fontWeight: 600, minWidth: 90 }}>Campaign:</span>
              <span>{linkedCampaign.title || linkedCampaign.campaign_id}</span>
            </div>
          )}
          {leaderboardEntry && (
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <span style={{ color: 'var(--text-muted)', fontWeight: 600, minWidth: 90 }}>Leaderboard:</span>
              <TierBadge tier={leaderboardEntry.tier} entry={leaderboardEntry} />
              <span style={{ fontWeight: 600, color: 'var(--accent-green)' }}>
                {Number(leaderboardEntry.composite_score).toFixed(3)}
              </span>
              {leaderboardEntry.tier === 'screening' && <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>needs investigation to advance</span>}
              {leaderboardEntry.tier === 'investigation' && !leaderboardEntry.investigation_passed && (
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>investigation completed (below threshold)</span>
              )}
              {leaderboardEntry.tier === 'investigation' && leaderboardEntry.investigation_passed && (
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>ready for validation</span>
              )}
            </div>
          )}
          <div style={{ display: 'flex', gap: 6, marginTop: 4, flexWrap: 'wrap' }}>
            {program?.experiment_id && onSelectExperiment && (
              <button className="refresh-btn" style={{ fontSize: 11, padding: '3px 10px' }} onClick={() => { onClose(); onSelectExperiment(program.experiment_id); }}>
                Open Experiment
              </button>
            )}
            {onAddToComparison && (
              <button className="refresh-btn" title="Add to side-by-side comparison" onClick={() => onAddToComparison(resultId)}>
                Compare
              </button>
            )}
            {onOpenInDesigner && (
              <button className="refresh-btn" title="Open this architecture in Aria Designer" onClick={() => onOpenInDesigner(resultId)}>
                Open in Designer
              </button>
            )}
            {leaderboardEntry && onViewInLeaderboard && (
              <button className="refresh-btn" style={{ fontSize: 11, padding: '3px 10px' }} onClick={() => { onClose(); onViewInLeaderboard(resultId); }}>
                View in Leaderboard
              </button>
            )}
            {linkedCampaign && onSelectCampaign && (
              <button className="refresh-btn" style={{ fontSize: 11, padding: '3px 10px' }} onClick={() => { onClose(); onSelectCampaign(linkedCampaign.campaign_id); }}>
                Open Campaign
              </button>
            )}
            <button
              className="refresh-btn"
              style={{ fontSize: 11, padding: '3px 10px', background: decisionPacket ? 'rgba(188, 140, 255, 0.15)' : undefined, borderColor: 'var(--accent-purple)', color: 'var(--accent-purple)' }}
              disabled={decisionPacketLoading}
              onClick={fetchDecisionPacket}
            >
              {decisionPacketLoading ? 'Loading...' : 'Decision Packet'}
            </button>
            <button className="refresh-btn" style={{ fontSize: 11, padding: '3px 10px' }} disabled={manifestLoading} onClick={fetchAndCopyManifest}>
              {manifestLoading ? 'Loading...' : manifestCopied ? 'Copied!' : 'Copy Manifest'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export function DecisionPacketPanel({ decisionPacket, decisionPacketError, decisionPacketOpen, setDecisionPacketOpen }) {
  if (!decisionPacket && !decisionPacketError) return null;
  return (
    <>
      {decisionPacketError && (
        <div style={{ padding: 8, background: 'rgba(248, 81, 73, 0.1)', border: '1px solid var(--accent-red)', borderRadius: 4, fontSize: 12, color: 'var(--accent-red)' }}>
          {decisionPacketError}
        </div>
      )}
      {decisionPacket && (
        <div style={{ background: 'var(--bg-tertiary)', borderRadius: 6, border: '1px solid var(--accent-purple)', overflow: 'hidden' }}>
          <div
            onClick={() => setDecisionPacketOpen(!decisionPacketOpen)}
            style={{ padding: '8px 12px', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center', userSelect: 'none', background: 'rgba(188, 140, 255, 0.08)' }}
          >
            <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--accent-purple)', textTransform: 'uppercase' }}>
              Decision Packet
            </span>
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              {decisionPacketOpen ? '▾ collapse' : '▸ expand'}
            </span>
          </div>
          {decisionPacketOpen && (
            <div style={{ padding: '8px 12px 12px', display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>Evidence Flags</div>
                <EvidenceFlagChips flags={decisionPacket.evidence_flags} />
              </div>
              <div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>Hypothesis Lineage</div>
                <HypothesisLineage chain={decisionPacket.hypothesis_chain} />
              </div>
              <div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>Outcomes by Phase</div>
                <OutcomesByPhase outcomes={decisionPacket.outcomes} />
              </div>
              {(decisionPacket.failure_context?.stage_at_death || decisionPacket.failure_context?.error_type) && (
                <div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>Failure Context</div>
                  <FailureContext context={decisionPacket.failure_context} />
                </div>
              )}
              <div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>Recommendation</div>
                <RecommendationCard recommendation={decisionPacket.recommendation} />
              </div>
            </div>
          )}
        </div>
      )}
    </>
  );
}

export function RefinementTracePanel({
  latestRefineLaunch,
  labels,
  shortId,
  onClose,
  onSelectExperiment,
  onViewInLeaderboard,
  refineTraceLoading,
  refineTrace,
  fmtInt,
  fmt,
  refineLaunchHistory,
}) {
  if (!latestRefineLaunch) return null;
  const resolvedLabels = labels || {
    panelTitle: 'Refinement Trace',
    openRefinementRun: 'Open Refinement Run',
    viewTopRefinedResult: 'View Top Refined Result',
    newFingerprints: 'New Fingerprints',
    openFingerprint: 'Open Fingerprint',
    recentRefinementLaunches: 'Recent Refinement Launches',
  };
  return (
    <div style={{ padding: 10, background: 'var(--bg-tertiary)', borderRadius: 6, border: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', color: 'var(--accent-purple)' }}>{resolvedLabels.panelTitle}</div>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'grid', gap: 4 }}>
        <div><strong>Intent:</strong> {latestRefineLaunch.intent}</div>
        <div><strong>Experiment:</strong> <span style={{ fontFamily: 'monospace' }}>{shortId(latestRefineLaunch.experimentId, 16)}</span></div>
        <div><strong>Source:</strong> <span style={{ fontFamily: 'monospace' }}>{shortId(latestRefineLaunch.sourceResultId, 12)}</span> · {shortId(latestRefineLaunch.sourceFingerprint, 18)}</div>
        <div><strong>Resolved IDs:</strong> {latestRefineLaunch.resolvedResultIds.length > 0 ? latestRefineLaunch.resolvedResultIds.map((v) => shortId(v, 10)).join(', ') : 'none'}</div>
        {latestRefineLaunch.unresolvedFingerprints.length > 0 && (
          <div><strong>Unresolved fingerprints:</strong> {latestRefineLaunch.unresolvedFingerprints.join(', ')}</div>
        )}
      </div>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        {latestRefineLaunch.experimentId && onSelectExperiment && (
          <button className="refresh-btn" style={{ fontSize: 11, padding: '3px 10px' }} onClick={() => { onClose(); onSelectExperiment(latestRefineLaunch.experimentId); }}>
            {resolvedLabels.openRefinementRun}
          </button>
        )}
        {refineTrace?.newResultIds?.[0] && onViewInLeaderboard && (
          <button className="refresh-btn" style={{ fontSize: 11, padding: '3px 10px' }} onClick={() => { onClose(); onViewInLeaderboard(refineTrace.newResultIds[0]); }}>
            {resolvedLabels.viewTopRefinedResult}
          </button>
        )}
      </div>
      {refineTraceLoading && <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Collecting live refinement outcomes…</div>}
      {refineTrace?.error && <div style={{ fontSize: 11, color: 'var(--accent-red)' }}>{refineTrace.error}</div>}
      {refineTrace && !refineTrace.error && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
          <div><strong>Status:</strong> {refineTrace.status}</div>
          <div><strong>Outcomes:</strong> {fmtInt(refineTrace.totals?.programs)} programs, {fmtInt(refineTrace.totals?.stage1Survivors)} S1 survivors, best loss {fmt(refineTrace.totals?.bestLoss)}</div>
          {refineTrace.newFingerprints?.length > 0 && (
            <div><strong>{resolvedLabels.newFingerprints}:</strong> {refineTrace.newFingerprints.map((fp) => shortId(fp, 18)).join(', ')}</div>
          )}
          {refineTrace.newCandidates?.length > 0 && (
            <div>
              <strong>{resolvedLabels.openFingerprint}:</strong>{' '}
              {onViewInLeaderboard ? (
                <span style={{ display: 'inline-flex', gap: 6, flexWrap: 'wrap', marginTop: 4 }}>
                  {refineTrace.newCandidates.map((candidate) => (
                    <button
                      key={candidate.resultId}
                      className="refresh-btn"
                      style={{ fontSize: 10, padding: '2px 8px', fontFamily: 'monospace' }}
                      onClick={() => { onClose(); onViewInLeaderboard(candidate.resultId); }}
                      title={`Open ${candidate.fingerprint}`}
                    >
                      {shortId(candidate.fingerprint, 18)}
                    </button>
                  ))}
                </span>
              ) : (
                refineTrace.newCandidates.map((candidate) => shortId(candidate.fingerprint, 18)).join(', ')
              )}
            </div>
          )}
          {refineTrace.newResultIds?.length > 0 && (
            <div><strong>New Result IDs:</strong> {refineTrace.newResultIds.map((rid) => shortId(rid, 10)).join(', ')}</div>
          )}
        </div>
      )}
      {refineLaunchHistory.length > 0 && (
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: 8 }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6 }}>
            {resolvedLabels.recentRefinementLaunches}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {refineLaunchHistory.map((item) => (
              <div key={item.experimentId} style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', fontSize: 11 }}>
                <span style={{ color: 'var(--text-secondary)' }}>{item.intent}</span>
                <span style={{ fontFamily: 'monospace', color: 'var(--text-muted)' }}>{shortId(item.experimentId, 12)}</span>
                <span style={{ color: 'var(--text-muted)' }}>{item.status || 'running'}</span>
                {onSelectExperiment && (
                  <button className="refresh-btn" style={{ fontSize: 10, padding: '2px 8px' }} onClick={() => { onClose(); onSelectExperiment(item.experimentId); }}>
                    Open Run
                  </button>
                )}
                {item.topCandidate?.resultId && onViewInLeaderboard && (
                  <button
                    className="refresh-btn"
                    style={{ fontSize: 10, padding: '2px 8px', fontFamily: 'monospace' }}
                    onClick={() => { onClose(); onViewInLeaderboard(item.topCandidate.resultId); }}
                    title={`Open ${item.topCandidate.fingerprint}`}
                  >
                    Open Fingerprint
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
