// Script to patch Discoveries.js
const fs = require('fs');
const file = 'research/dashboard/src/components/Discoveries.js';
let code = fs.readFileSync(file, 'utf8');

// The main loop looks like this:
//              {filtered.map((entry, i) => {

let loopRegex = /\{\s*filtered\.map\(\s*\(\s*entry\s*,\s*i\s*\)\s*=>\s*\{([\s\S]*?)return\s*\(\s*<React\.Fragment[\s\S]*?<\/React\.Fragment>\s*\)\s*;\s*\}\)\s*\}/m;

const match = loopRegex.exec(code);
if (!match) {
  console.log("Could not find filtered.map block!");
  process.exit(1);
}

const rowComponentDef = `
const DiscoveryRow = React.memo(function DiscoveryRow({
  entry, 
  i, 
  rowId, 
  isExpanded, 
  isHighlighted, 
  isQueued, 
  isPinnedReference, 
  eligibility, 
  displayName,
  highlightRef,
  onSelectProgram,
  tdStyle,
  COLUMNS,
  visibleColumns,
  onOpenInDesigner,
  setExpandedRowId,
  actionBtnStyle,
  handleDelete,
  onInvestigate,
  onValidate,
  onQueueAdd,
  onQueueRemove,
  statusDrafts,
  handleStatusDraftChange,
  handleSaveStatus,
  savingStatusRowId
}) {
  return (
    <React.Fragment>
      <tr
        ref={isHighlighted ? highlightRef : undefined}
        style={{
          borderBottom: '1px solid var(--border)',
          cursor: 'pointer',
          background: isHighlighted
            ? 'rgba(88, 166, 255, 0.2)'
            : isPinnedReference
              ? 'rgba(188, 140, 255, 0.14)'
              : entry.tier === 'breakthrough' ? 'rgba(63, 185, 80, 0.08)' : undefined,
          animation: isHighlighted ? 'leaderboard-pulse 1.5s ease-in-out 2' : undefined,
        }}
        onClick={() => onSelectProgram?.(entry.result_id)}
      >
        <td style={{ ...tdStyle, width: 26, textAlign: 'center', paddingLeft: 4, paddingRight: 4 }}>
          {isPinnedReference ? (
            <span title="Pinned reference" style={{ color: 'var(--accent-purple)', fontSize: 12, fontWeight: 700 }}>
              ★
            </span>
          ) : null}
        </td>
        <td style={tdStyle}>{i + 1}</td>
        {COLUMNS.filter(col => visibleColumns.includes(col.key)).map(col => {
          switch (col.key) {
            case '_score':
              return <td key={col.key} style={tdStyle}><ScoreCell entry={entry} /></td>;
            case 'display_name':
              return (
                <td key={col.key} style={{ ...tdStyle, maxWidth: 200 }}>
                  <div style={{ fontWeight: 500 }}>{displayName}</div>
                  {entry.graph_fingerprint && (
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                      {entry.graph_fingerprint.slice(0, 12)}
                    </div>
                  )}
                </td>
              );
            case 'architecture_family':
              return (
                <td key={col.key} style={tdStyle}>
                  <span style={{
                    fontSize: 11, padding: '1px 6px', borderRadius: 3,
                    background: 'var(--bg-tertiary)', color: 'var(--text-secondary)',
                  }}>
                    {entry.architecture_family || '--'}
                  </span>
                </td>
              );
            case 'discovery_loss_ratio':
              const discoveryDisplay = discoveryLossDisplay(entry);
              return (
                <td key={col.key} style={{ ...tdStyle, color: lossColor(discoveryDisplay), fontFamily: 'monospace' }}>
                  {discoveryDisplay != null ? Number(discoveryDisplay).toFixed(4) : '--'}
                </td>
              );
            case 'validation_loss_ratio':
              const validationDisplay = validationLossDisplay(entry);
              return (
                <td key={col.key} style={{ ...tdStyle, color: lossColor(validationDisplay), fontFamily: 'monospace' }}>
                  {validationDisplay != null ? Number(validationDisplay).toFixed(4) : '--'}
                </td>
              );
            case '_best_loss':
              return (
                <td key={col.key} style={{ ...tdStyle, color: lossColor(entry._best_loss), fontFamily: 'monospace' }}>
                  {entry._best_loss != null ? (Number(entry._best_loss) !== 0 && Math.abs(Number(entry._best_loss)) < 0.0001 ? Number(entry._best_loss).toExponential(2) : Number(entry._best_loss).toFixed(4)) : '--'}
                </td>
              );
            case '_vs_ref':
              return (
                <td key={col.key} style={{ ...tdStyle, fontFamily: 'monospace', color: entry._vs_ref != null ? (entry._vs_ref <= 100 ? 'var(--accent-green)' : 'var(--accent-red)') : 'var(--text-muted)' }}>
                  {entry._vs_ref != null ? \`\${entry._vs_ref.toFixed(1)}%\` : '--'}
                </td>
              );
            case '_novelty':
              return (
                <td key={col.key} style={{ ...tdStyle, color: noveltyColor(entry._novelty), fontFamily: 'monospace' }}>
                  {entry._novelty != null ? Number(entry._novelty).toFixed(3) : '--'}
                </td>
              );
            case 'param_efficiency':
              return (
                <td key={col.key} style={tdStyle}>{entry.param_efficiency != null ? Number(entry.param_efficiency).toFixed(3) : '--'}</td>
              );
            case 'robustness_long_ctx_best_score':
              return <td key={col.key} style={tdStyle}>{entry.robustness_long_ctx_best_score != null ? Number(entry.robustness_long_ctx_best_score).toFixed(3) : '--'}</td>;
            case 'robustness_long_ctx_multi_hop_score':
              return <td key={col.key} style={tdStyle}>{entry.robustness_long_ctx_multi_hop_score != null ? Number(entry.robustness_long_ctx_multi_hop_score).toFixed(3) : '--'}</td>;
            case 'robustness_long_ctx_passkey_score':
              return <td key={col.key} style={tdStyle}>{entry.robustness_long_ctx_passkey_score != null ? Number(entry.robustness_long_ctx_passkey_score).toFixed(3) : '--'}</td>;
            case 'robustness_long_ctx_retrieval_aggregate':
              return <td key={col.key} style={tdStyle}>{entry.robustness_long_ctx_retrieval_aggregate != null ? Number(entry.robustness_long_ctx_retrieval_aggregate).toFixed(3) : '--'}</td>;
            case 'max_viable_seq_len':
              return <td key={col.key} style={tdStyle}>{entry.max_viable_seq_len != null ? Number(entry.max_viable_seq_len).toFixed(0) : '--'}</td>;
            case 'jacobian_spectral_norm':
              const specVal = finitePositiveOrNull(entry.jacobian_spectral_norm ?? entry.fp_jacobian_spectral_norm);
              return <td key={col.key} style={tdStyle}>{specVal != null ? Number(specVal).toFixed(4) : '--'}</td>;
            case 'init_sensitivity_std':
              return <td key={col.key} style={tdStyle}>{entry.init_sensitivity_std != null ? Number(entry.init_sensitivity_std).toFixed(4) : '--'}</td>;
            case 'tier':
              return (
                <td key={col.key} style={tdStyle}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <StatusBadge entry={entry} />
                  </div>
                </td>
              );
            case '_actions':
              return (
                <td key={col.key} style={tdStyle} onClick={e => e.stopPropagation()}>
                  <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                    <button
                      onClick={() => setExpandedRowId(isExpanded ? null : rowId)}
                      style={{
                        ...actionBtnStyle,
                        borderColor: 'var(--accent-blue)',
                        color: 'var(--accent-blue)',
                        background: isExpanded ? 'rgba(88, 166, 255, 0.12)' : 'transparent',
                      }}
                    >
                      {isExpanded ? 'Collapse' : 'Details'}
                    </button>
                    {onOpenInDesigner && (
                      <button
                        onClick={() => {
                          if (entry.result_id) onOpenInDesigner(entry.result_id)
                        }}
                        disabled={!entry.result_id}
                        style={{
                          ...actionBtnStyle,
                          borderColor: 'var(--accent-purple)',
                          color: 'var(--accent-purple)',
                          opacity: entry.result_id ? 1 : 0.5,
                          cursor: entry.result_id ? 'pointer' : 'not-allowed',
                        }}
                        title={entry.result_id ? 'Open architecture in visual designer' : 'Designer unavailable: missing result ID'}
                      >
                        Designer
                      </button>
                    )}
                    <button
                      onClick={() => {
                        if (window.confirm(\`Delete \${entry.entry_id?.slice(0, 11) || entry.result_id?.slice(0, 11)} and all associated data?\`)) {
                          handleDelete(entry.entry_id || entry.result_id);
                        }
                      }}
                      style={{
                        ...actionBtnStyle,
                        borderColor: 'rgba(248, 81, 73, 0.4)',
                        background: 'rgba(248, 81, 73, 0.12)',
                        color: 'var(--accent-red, #f85149)',
                      }}
                      title="Delete entry and all associated data"
                    >
                      Delete
                    </button>
                  </div>
                </td>
              );
            default:
              return null;
          }
        })}
      </tr>
      {isExpanded && (
        <ExpandedDetail
          entry={entry}
          onInvestigate={onInvestigate}
          onValidate={onValidate}
          onQueueAdd={onQueueAdd}
          onQueueRemove={onQueueRemove}
          onDelete={handleDelete}
          isQueued={isQueued}
          eligibility={eligibility}
          statusDraft={statusDrafts[rowId] || entry.tier}
          onStatusDraftChange={(tier) => handleStatusDraftChange(rowId, tier)}
          onSaveStatus={() => handleSaveStatus(entry)}
          savingStatus={savingStatusRowId === rowId}
        />
      )}
    </React.Fragment>
  );
});
`;

let replacementLoopBlock = `{filtered.map((entry, i) => {
                const rowId = entry.entry_id || entry.result_id || i;
                const isExpanded = expandedRowId === rowId;
                const isHighlighted = highlightId && entry.result_id === highlightId;
                const isQueued = !!entry.result_id && queuedSet.has(entry.result_id);
                const isPinnedReference = isPinnedReferenceRow(entry);
                const eligibility = eligibilityByResultId?.[entry.result_id] || null;
                const displayName = entry.display_name || entry.architecture_desc || entry.graph_fingerprint?.slice(0, 10) || '--';
                return (
                  <DiscoveryRow 
                    key={rowId}
                    entry={entry}
                    i={i}
                    rowId={rowId}
                    isExpanded={isExpanded}
                    isHighlighted={isHighlighted}
                    isQueued={isQueued}
                    isPinnedReference={isPinnedReference}
                    eligibility={eligibility}
                    displayName={displayName}
                    highlightRef={highlightRef}
                    onSelectProgram={onSelectProgram}
                    tdStyle={tdStyle}
                    COLUMNS={COLUMNS}
                    visibleColumns={visibleColumns}
                    onOpenInDesigner={onOpenInDesigner}
                    setExpandedRowId={setExpandedRowId}
                    actionBtnStyle={actionBtnStyle}
                    handleDelete={handleDelete}
                    onInvestigate={onInvestigate}
                    onValidate={onValidate}
                    onQueueAdd={onQueueAdd}
                    onQueueRemove={onQueueRemove}
                    statusDrafts={statusDrafts}
                    handleStatusDraftChange={handleStatusDraftChange}
                    handleSaveStatus={handleSaveStatus}
                    savingStatusRowId={savingStatusRowId}
                  />
                );
              })}`;

code = code.replace(loopRegex, replacementLoopBlock);
// Inject definition at end of file before `export default`
code = code.replace(/export default Discoveries;/, rowComponentDef + "\nexport default Discoveries;");

fs.writeFileSync(file, code);
