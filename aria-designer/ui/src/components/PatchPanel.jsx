import { memo, useState } from 'react'

function PatchPanel({ proposals, onApply, onReject, onPreview, onClose }) {
  const [selectedPatchId, setSelectedPatchId] = useState(null)
  
  const selectedPatch = proposals.find(p => p.id === selectedPatchId)

  const handleSelect = (patch) => {
    setSelectedPatchId(patch.id)
    if (onPreview) onPreview(patch)
  }

  return (
    <div className="patch-panel">
      <div className="panel-header">
        <h2>Aria Proposals</h2>
        {onClose && <button className="close-btn" onClick={onClose}>&times;</button>}
      </div>

      {proposals.length === 0 && (
        <p className="muted">No pending proposals from Aria.</p>
      )}

      <div className="patch-list">
        {proposals.map((patch) => (
          <div 
            key={patch.id} 
            className={`patch-item ${selectedPatchId === patch.id ? 'active' : ''}`}
            onClick={() => handleSelect(patch)}
          >
            <div className="patch-meta">
              <span className="patch-author">Aria</span>
              <span className="patch-date">{new Date(patch.created_at).toLocaleTimeString()}</span>
            </div>
            <div className="patch-rationale">{patch.rationale}</div>
          </div>
        ))}
      </div>

      {selectedPatch && (
        <div className="patch-details">
          <h3>Proposal Details</h3>
          <div className="rationale-full">{selectedPatch.rationale}</div>
          
          <div className="ops-list">
            <h4>Operations</h4>
            {JSON.parse(selectedPatch.patch_json).ops.map((op, idx) => (
              <div key={idx} className="op-item">
                <span className="op-type">{op.op}</span>
                <span className="op-target">{op.node_id || op.edge_id || ''}</span>
              </div>
            ))}
          </div>

          <div className="patch-actions">
            <button className="approve-btn" onClick={() => onApply(selectedPatch.id)}>Apply Patch</button>
            <button className="reject-btn" onClick={() => onReject(selectedPatch.id)}>Reject</button>
          </div>
        </div>
      )}
    </div>
  )
}

export default memo(PatchPanel)
