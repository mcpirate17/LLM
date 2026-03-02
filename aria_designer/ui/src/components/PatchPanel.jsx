import { memo, useEffect, useState } from 'react'

function PatchPanel({ proposals, onApply, onReject, onPreview, onClose }) {
  const [selectedPatchId, setSelectedPatchId] = useState(null)
  
  useEffect(() => {
    if (proposals.length === 0) {
      setSelectedPatchId(null)
      return
    }
    if (!selectedPatchId || !proposals.some((p) => p.id === selectedPatchId)) {
      const first = proposals[0]
      setSelectedPatchId(first.id)
      if (onPreview) onPreview(first)
    }
  }, [proposals, selectedPatchId, onPreview])
  
  const selectedPatch = proposals.find(p => p.id === selectedPatchId)
  let parsedOps = []
  let patchParseError = null
  if (selectedPatch?.patch_json) {
    try {
      const parsed = JSON.parse(selectedPatch.patch_json)
      parsedOps = Array.isArray(parsed?.ops) ? parsed.ops : []
    } catch {
      patchParseError = 'Patch payload is invalid JSON'
    }
  }

  const handleSelect = (patch) => {
    setSelectedPatchId(patch.id)
    if (onPreview) onPreview(patch)
  }

  return (
    <div className="patch-panel">
      <div className="panel-header">
        <h2>Aria Proposals</h2>
        {onClose && <button type="button" className="close-btn" onClick={onClose}>&times;</button>}
      </div>

      {proposals.length === 0 && (
        <p className="muted">No pending proposals from Aria.</p>
      )}

      <div className="patch-list">
        {proposals.map((patch) => {
          let opCount = 0
          try {
            const parsed = JSON.parse(patch.patch_json || '{}')
            opCount = Array.isArray(parsed?.ops) ? parsed.ops.length : 0
          } catch {}
          return (
            <button
              type="button"
              key={patch.id}
              className={`patch-item ${selectedPatchId === patch.id ? 'active' : ''}`}
              aria-pressed={selectedPatchId === patch.id}
              onClick={() => handleSelect(patch)}
            >
              <div className="patch-meta">
                <span className="patch-author">Aria</span>
                <span className="patch-date">{new Date(patch.created_at).toLocaleTimeString()}</span>
              </div>
              <div className="patch-rationale">{patch.rationale}</div>
              <div className="patch-badges">
                <span className="patch-badge">{opCount} op{opCount === 1 ? '' : 's'}</span>
                <span className="patch-badge">id: {String(patch.id || '').slice(0, 10)}</span>
              </div>
            </button>
          )
        })}
      </div>

      {selectedPatch && (
        <div className="patch-details">
          <h3>Proposal Details</h3>
          <div className="rationale-full">{selectedPatch.rationale}</div>
          
          <div className="ops-list">
            <h4>Operations</h4>
            {patchParseError ? (
              <div className="muted">{patchParseError}</div>
            ) : parsedOps.length === 0 ? (
              <div className="muted">No operations found in this proposal.</div>
            ) : parsedOps.map((op, idx) => (
              <div key={idx} className="op-item">
                <span className="op-type">{op.op}</span>
                <span className="op-target">{op.node_id || op.edge_id || ''}</span>
              </div>
            ))}
          </div>

          <div className="patch-actions">
            <button type="button" className="approve-btn" onClick={() => onApply(selectedPatch.id)} disabled={Boolean(patchParseError)}>Apply Patch</button>
            <button type="button" className="reject-btn" onClick={() => onReject(selectedPatch.id)}>Reject</button>
          </div>
        </div>
      )}
    </div>
  )
}

export default memo(PatchPanel)
