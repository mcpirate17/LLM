import { useCallback, useRef, useState } from 'react'
import { apiCall } from '../services/apiService'
import { buildWorkflowJson } from '../utils/workflow'

/**
 * Aria co-design: suggest, submit prompt, apply/reject/preview patches, ghost click.
 */
export function useAriaCoDesign({
  nodes, edges, workflowMeta, evalState,
  setNodes, setStatusMsg, setRightPanelTab,
  loadWorkflowJsonRef, pushToast, proposalQuery,
}) {
  const [ariaSuggestions, setAriaSuggestions] = useState([])
  const [ariaLoading, setAriaLoading] = useState(false)
  const [proposals, setProposals] = useState([])
  const [previewPatch, setPreviewPatch] = useState(null)
  const [showAskAriaModal, setShowAskAriaModal] = useState(false)
  const handleApplyPatchRef = useRef(null)

  const handleAskAriaSuggest = useCallback(async (promptText = '') => {
    const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
    const prompt = String(promptText || '').trim()
    setAriaLoading(true)
    setStatusMsg('Fetching Aria suggestions...')
    try {
      const res = await apiCall(`/api/v1/aria/suggest-components`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflow, prompt: prompt || undefined }),
      })
      const data = await res.json()
      const suggestions = Array.isArray(data) ? data : []
      setAriaSuggestions(suggestions)
      setStatusMsg(`Aria suggested ${suggestions.length} component(s)`)
    } catch {
      setStatusMsg('Failed to fetch Aria suggestions')
    } finally {
      setAriaLoading(false)
    }
  }, [nodes, edges, workflowMeta, setStatusMsg])

  const handleAskAriaSubmit = useCallback(async (promptText, options = {}) => {
    const autoApply = Boolean(options.autoApply)
    const prompt = String(promptText || '').trim()
    if (!prompt) return
    const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
    const benchmarkSummary = evalState?.benchmarking?.summary || null
    const offTarget = Array.isArray(evalState?.benchmarking?.targets)
      ? evalState.benchmarking.targets.filter((t) => t.status === 'off_target').slice(0, 4)
      : []
    const benchmarkContext = benchmarkSummary
      ? `Benchmark context: score ${(benchmarkSummary.score ?? 0).toFixed(2)} with ${benchmarkSummary.on_target || 0} on-target and ${benchmarkSummary.off_target || 0} off-target metrics.`
      : ''
    const offTargetContext = offTarget.length > 0
      ? `Highest-priority gaps: ${offTarget.map((t) => `${t.label} (observed=${t.observed ?? 'n/a'}, target=${t.target})`).join('; ')}.`
      : ''
    const effectivePrompt = [benchmarkContext, offTargetContext, prompt].filter(Boolean).join(' ')
    setAriaLoading(true)
    setStatusMsg('Generating Aria patch...')

    try {
      const genRes = await apiCall(`/api/v1/aria/generate-patch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflow, prompt: effectivePrompt, base_version: 1 }),
      })
      const genData = await genRes.json()
      if (!genRes.ok || !genData.proposal_id) {
        throw new Error(genData.detail || genData.error || 'generate-patch failed')
      }

      setStatusMsg('Applying patch...')
      const applyRes = await apiCall(`/api/v1/aria/apply-patch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ proposal_id: genData.proposal_id, approved_by: 'user' }),
      })
      const applyData = await applyRes.json().catch(() => ({}))
      if (applyRes.ok && applyData.applied && applyData.patched_workflow) {
        setShowAskAriaModal(false)
        loadWorkflowJsonRef.current?.(applyData.patched_workflow)
        const fp = applyData.new_fingerprint
          ? ` (fingerprint: ${applyData.new_fingerprint.slice(0, 8)}\u2026)`
          : ''
        setStatusMsg(`Patch applied: ${applyData.ops_applied} operations${fp}`)
        setAriaLoading(false)
        return
      }

      if (!autoApply) {
        setStatusMsg(`Aria proposal created: ${genData.proposal_id}`)
        setRightPanelTab('proposals')
        setShowAskAriaModal(false)
        if (proposalQuery) {
          const pRes = await apiCall(proposalQuery)
          const pData = await pRes.json()
          setProposals(Array.isArray(pData) ? pData : [])
        }
        setAriaLoading(false)
        return
      }

      const errDetail = applyData.detail || applyData.error || 'Apply returned no patched workflow'
      throw new Error(errDetail)
    } catch (err) {
      console.error('[Ask Aria]', err)
      setStatusMsg(`Aria patch failed: ${err.message || err}`)
      setAriaLoading(false)
    }
  }, [nodes, edges, workflowMeta, evalState, proposalQuery, setStatusMsg, setRightPanelTab, loadWorkflowJsonRef])

  const handleApplyPatch = useCallback(async (proposalId) => {
    setStatusMsg('Applying patch...')
    try {
      const res = await apiCall(`/api/v1/aria/apply-patch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ proposal_id: proposalId, approved_by: 'user' }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok || !data.applied) {
        throw new Error(data.detail || data.error || 'Patch application failed')
      }
      const fp = data.new_fingerprint
        ? ` (fingerprint: ${data.new_fingerprint.slice(0, 8)}\u2026)`
        : ''
      setStatusMsg(`Patch applied: ${data.ops_applied} operations, saved as v${data.new_version}${fp}`)
      setProposals((prev) => prev.filter(p => p.id !== proposalId))

      if (data.patched_workflow) {
        loadWorkflowJsonRef.current?.(data.patched_workflow)
      }
    } catch (err) {
      const msg = String(err?.message || err || 'Patch application failed')
      if (msg.toLowerCase().includes('proposal is stale') || msg.includes('HTTP 409')) {
        try {
          await apiCall(`/api/v1/aria/reject-patch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ proposal_id: proposalId, approved_by: 'user' }),
          })
        } catch {}
        setProposals((prev) => prev.filter(p => p.id !== proposalId))
        setStatusMsg('Proposal was stale and has been removed. Generate a new proposal on the latest graph.')
        return
      }
      setStatusMsg(`Failed to apply patch: ${msg}`)
    }
  }, [setStatusMsg, loadWorkflowJsonRef])

  handleApplyPatchRef.current = handleApplyPatch

  const handleRejectPatch = useCallback(async (proposalId) => {
    try {
      await apiCall(`/api/v1/aria/reject-patch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ proposal_id: proposalId, approved_by: 'user' }),
      })
      setProposals((prev) => prev.filter(p => p.id !== proposalId))
      setStatusMsg('Patch rejected')
      setPreviewPatch(null)
    } catch {
      setStatusMsg('Failed to reject patch')
    }
  }, [setStatusMsg])

  const handlePreviewPatch = useCallback((patch) => {
    setPreviewPatch(patch)
    let ops = []
    try {
      ops = (JSON.parse(patch.patch_json).ops || [])
    } catch {
      setStatusMsg('Proposal preview unavailable: invalid patch payload')
    }
    const affectedNodeIds = ops.filter(op => op.node_id).map(op => op.node_id)
    setNodes(nds => nds.map(n => ({
      ...n,
      className: affectedNodeIds.includes(n.id) ? 'patch-preview-highlight' : ''
    })))
  }, [setNodes, setStatusMsg])

  const handleGhostClick = useCallback((suggestion) => {
    const c = suggestion.component
    const componentType = c.id.includes('/') ? c.id : `${c.category || 'math'}/${c.id}`
    const ghostNode = nodes.find(n => n.id === `ghost_${suggestion.id}`)
    const position = ghostNode ? ghostNode.position : { x: 400, y: 300 }

    const newNodeId = `aria_${Date.now().toString(36)}`
    const ops = [{
      op: 'add_node',
      payload: {
        id: newNodeId,
        component_type: componentType,
        params: suggestion.params || {},
        ui_meta: { position },
      }
    }]

    apiCall(`/api/v1/workflows/${workflowMeta.workflow_id}/patch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ operations: ops })
    })
    .then(res => res.json())
    .then(data => {
      if (data.workflow) {
        loadWorkflowJsonRef.current?.(data.workflow)
        setAriaSuggestions(prev => prev.filter(s => s.id !== suggestion.id))
        setStatusMsg(`Integrated Aria's ${c.name} suggestion`)
        pushToast(`Integrated Aria's ${c.name} suggestion`, 'success')
      }
    })
    .catch(err => {
      setStatusMsg(`Failed to integrate suggestion: ${err.message}`)
    })
  }, [nodes, workflowMeta, pushToast, setStatusMsg, loadWorkflowJsonRef])

  return {
    ariaSuggestions, setAriaSuggestions,
    ariaLoading,
    proposals, setProposals,
    previewPatch, setPreviewPatch,
    showAskAriaModal, setShowAskAriaModal,
    handleApplyPatchRef,
    handleAskAriaSuggest,
    handleAskAriaSubmit,
    handleApplyPatch,
    handleRejectPatch,
    handlePreviewPatch,
    handleGhostClick,
  }
}
