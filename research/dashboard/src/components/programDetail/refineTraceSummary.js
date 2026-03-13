export function summarizeRefineTrace(payload, sourceResultId, currentFingerprint) {
  const experiment = payload?.experiment || {};
  const programs = Array.isArray(payload?.programs) ? payload.programs : [];

  const withRefinementMeta = programs.map((row) => {
    let refinement = null;
    try {
      const raw = row?.graph_json;
      if (raw && typeof raw === 'string') {
        const parsed = JSON.parse(raw);
        refinement = parsed?.metadata?.refinement || null;
      }
    } catch (_) {
      refinement = null;
    }
    return { ...row, _refinement: refinement };
  });

  const lineage = withRefinementMeta.filter(
    (row) => String(row?._refinement?.source_result_id || '') === String(sourceResultId),
  );
  const scoped = lineage.length > 0 ? lineage : withRefinementMeta;

  const finiteLosses = scoped
    .map((row) => Number(row?.loss_ratio))
    .filter((value) => Number.isFinite(value));
  const bestLoss = finiteLosses.length > 0 ? Math.min(...finiteLosses) : null;
  const stage1Survivors = scoped.filter((row) => Boolean(row?.stage1_passed)).length;

  const uniqueFingerprints = [];
  const uniqueResultIds = [];
  const newCandidates = [];
  for (const row of scoped) {
    const fp = String(row?.graph_fingerprint || '').trim();
    const rid = String(row?.result_id || '').trim();
    if (fp && fp !== String(currentFingerprint || '') && !uniqueFingerprints.includes(fp)) {
      uniqueFingerprints.push(fp);
    }
    if (rid && rid !== String(sourceResultId) && !uniqueResultIds.includes(rid)) {
      uniqueResultIds.push(rid);
    }
    if (rid && fp && rid !== String(sourceResultId) && !newCandidates.some((candidate) => candidate.resultId === rid)) {
      newCandidates.push({ resultId: rid, fingerprint: fp });
    }
  }

  const status = String(experiment?.status || '').toLowerCase();
  const completed = Boolean(experiment?.completed_at) || status === 'completed' || status === 'failed' || status === 'cancelled';

  return {
    status: status || 'running',
    completed,
    experiment,
    totals: {
      programs: programs.length,
      scopedPrograms: scoped.length,
      stage1Survivors,
      bestLoss,
    },
    newFingerprints: uniqueFingerprints.slice(0, 6),
    newResultIds: uniqueResultIds.slice(0, 6),
    newCandidates: newCandidates.slice(0, 6),
  };
}
