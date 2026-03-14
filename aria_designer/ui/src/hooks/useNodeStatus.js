import { useCallback } from 'react'

/**
 * Node highlighting, error collection, and eval status management.
 * @param {Array} nodes - current nodes array
 * @param {Function} setNodes - React Flow setNodes
 * @returns {{ clearNodeHighlights, highlightNodeErrors, collectFailureErrorMap, setAllNodeEvalStatus }}
 */
export function useNodeStatus(nodes, setNodes) {
  const clearNodeHighlights = useCallback(() => {
    setNodes((nds) => nds.map((n) => ({ ...n, className: '', data: { ...n.data, errors: [] } })))
  }, [setNodes])

  const resolveNodeIdsFromErrorText = useCallback((text) => {
    const msg = String(text || '').trim()
    if (!msg) return []
    const lower = msg.toLowerCase()
    const matches = new Set()
    for (const n of nodes) {
      const id = String(n.id || '')
      if (!id) continue
      const idLower = id.toLowerCase()
      const labelLower = String(n.data?.label || '').toLowerCase()
      const compLower = String(n.data?.componentId || '').toLowerCase()
      const compLeaf = compLower.split('/').pop() || compLower
      if (
        lower.includes(idLower)
        || (labelLower && lower.includes(labelLower))
        || (compLeaf && lower.includes(compLeaf))
      ) {
        matches.add(id)
      }
    }
    return [...matches]
  }, [nodes])

  const collectFailureErrorMap = useCallback((payload, fallbackMessage = '') => {
    const errorMap = {}
    const pushError = (nodeId, message) => {
      const id = String(nodeId || '').trim()
      if (!id) return
      if (!errorMap[id]) errorMap[id] = []
      errorMap[id].push(String(message || 'Execution failure'))
    }
    if (payload && typeof payload === 'object') {
      if (payload.node_statuses && typeof payload.node_statuses === 'object') {
        for (const [nodeId, info] of Object.entries(payload.node_statuses)) {
          if (info?.valid === false || (Array.isArray(info?.errors) && info.errors.length > 0)) {
            const errs = Array.isArray(info?.errors) && info.errors.length > 0 ? info.errors : ['Node failed']
            for (const err of errs) pushError(nodeId, err)
          }
        }
      }
      if (Array.isArray(payload.issues)) {
        for (const issue of payload.issues) {
          if (issue?.node_id) pushError(issue.node_id, issue.message || 'Validation issue')
        }
      }
      if (Array.isArray(payload.errors)) {
        for (const err of payload.errors) {
          if (err?.node_id) {
            pushError(err.node_id, err.message || err.error || 'Node failed')
            continue
          }
          const msg = String(err?.message || err?.error || err || '')
          for (const nid of resolveNodeIdsFromErrorText(msg)) pushError(nid, msg)
        }
      }
    }
    if (Object.keys(errorMap).length === 0 && fallbackMessage) {
      for (const nid of resolveNodeIdsFromErrorText(fallbackMessage)) {
        pushError(nid, fallbackMessage)
      }
    }
    return errorMap
  }, [resolveNodeIdsFromErrorText])

  const highlightNodeErrors = useCallback((errorMap) => {
    const keys = Object.keys(errorMap || {})
    if (keys.length === 0) return
    setNodes((nds) =>
      nds.map((n) => {
        const errs = errorMap[n.id]
        if (!errs || errs.length === 0) return n
        return {
          ...n,
          className: 'node-invalid',
          data: {
            ...n.data,
            errors: errs,
            evalStatus: 'fail',
            evalError: errs[0] || n.data?.evalError || null,
          },
        }
      })
    )
  }, [setNodes])

  const setAllNodeEvalStatus = useCallback((status, error) => {
    setNodes((nds) =>
      nds.map((n) => ({
        ...n,
        data: { ...n.data, evalStatus: status, evalError: error || null },
      }))
    )
  }, [setNodes])

  return { clearNodeHighlights, highlightNodeErrors, collectFailureErrorMap, setAllNodeEvalStatus }
}
