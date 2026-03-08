import { useCallback, useState, useEffect } from 'react'
import { apiCall } from "../services/apiService"
import { buildWorkflowJson } from '../utils/workflow'

export function useAria(nodes, edges, workflowMeta, setNodes) {
  const [proposals, setProposals] = useState([])
  const [ariaLoading, setAriaLoading] = useState(false)
  const [ariaSuggestions, setAriaSuggestions] = useState([])
  const [previewPatchId, setPreviewPatchId] = useState(null)

  useEffect(() => {
    apiCall('/api/v1/aria/proposals?status=pending')
      .then(r => r.json())
      .then(data => setProposals(Array.isArray(data) ? data : []))
      .catch(err => console.error("Failed to fetch proposals", err))
  }, [])

  const handleAskAriaSuggest = useCallback(async (promptText = '') => {
    setAriaLoading(true)
    try {
      const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
      const res = await apiCall('/api/v1/aria/suggest-components', {
        method: 'POST',
        body: { workflow, prompt: promptText || undefined },
      })
      const data = await res.json()
      setAriaSuggestions(Array.isArray(data) ? data : [])
    } catch (err) {
      console.error("Failed to fetch Aria suggestions", err)
    } finally {
      setAriaLoading(false)
    }
  }, [nodes, edges, workflowMeta])

  const handleAskAriaSubmit = useCallback(async (promptText) => {
    setAriaLoading(true)
    try {
      const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
      const res = await apiCall('/api/v1/aria/propose-patch', {
        method: 'POST',
        body: { 
          workflow_id: workflow.workflow_id,
          rationale: promptText,
          // In a real app, the backend would generate the ops from the prompt.
          // Here we just send the prompt and expect a proposal to be created.
          prompt: promptText 
        },
      })
      const data = await res.json()
      if (data.proposal_id) {
        // Refresh proposals
        const pRes = await apiCall('/api/v1/aria/proposals?status=pending')
        const pData = await pRes.json()
        setProposals(Array.isArray(pData) ? pData : [])
      }
    } catch (err) {
      console.error("Failed to submit Aria prompt", err)
    } finally {
      setAriaLoading(false)
    }
  }, [nodes, edges, workflowMeta])

  const handleApplyPatch = useCallback(async (proposalId) => {
    try {
      const res = await apiCall('/api/v1/aria/apply-patch', {
        method: 'POST',
        body: { proposal_id: proposalId },
      })
      const data = await res.json()
      if (data.status === 'applied') {
        setProposals(prev => prev.filter(p => p.id !== proposalId))
        setPreviewPatchId(null)
        // Clear highlights
        setNodes(nds => nds.map(n => ({ ...n, className: '' })))
        return data.patched_workflow
      }
    } catch (err) {
      console.error("Failed to apply patch", err)
    }
    return null
  }, [setNodes])

  const handleRejectPatch = useCallback(async (proposalId) => {
    try {
      await apiCall(`/api/v1/aria/reject-patch?proposal_id=${proposalId}`, { method: 'POST' })
      setProposals(prev => prev.filter(p => p.id !== proposalId))
      if (previewPatchId === proposalId) {
        setPreviewPatchId(null)
        setNodes(nds => nds.map(n => ({ ...n, className: '' })))
      }
    } catch (err) {
      console.error("Failed to reject patch", err)
    }
  }, [previewPatchId, setNodes])

  const handlePreviewPatch = useCallback((proposal) => {
    if (previewPatchId === proposal.id) {
      setPreviewPatchId(null)
      setNodes(nds => nds.map(n => ({ ...n, className: '' })))
      return
    }

    setPreviewPatchId(proposal.id)
    const patch = typeof proposal.patch === 'string' ? JSON.parse(proposal.patch) : proposal.patch
    const ops = patch?.ops || []
    const affectedNodeIds = ops.filter(op => op.node_id).map(op => op.node_id)
    
    setNodes(nds => nds.map(n => ({
      ...n,
      className: affectedNodeIds.includes(n.id) ? 'patch-preview-highlight' : ''
    })))
  }, [previewPatchId, setNodes])

  return {
    proposals,
    ariaLoading,
    ariaSuggestions,
    handleAskAriaSuggest,
    handleAskAriaSubmit,
    handleApplyPatch,
    handleRejectPatch,
    handlePreviewPatch,
    previewPatchId
  }
}
