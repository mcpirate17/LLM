import { useCallback, useEffect, useRef, useState } from 'react'
import { apiCall } from '../services/apiService'
import { buildWorkflowJson } from '../utils/workflow'

export function useEmbeddedBridge({
  nodes, edges, workflowMeta,
  setStatusMsg, loadWorkflowJsonRef, formatImportError,
}) {
  const [embeddedMode, setEmbeddedMode] = useState(
    () => new URLSearchParams(window.location.search).get('embedded') === '1'
  )

  const postToParent = useCallback((type, payload = {}) => {
    if (!embeddedMode || !window.parent || window.parent === window) return
    window.parent.postMessage({ source: 'aria_designer', type, ...payload }, '*')
  }, [embeddedMode])

  const importingRef = useRef(null)
  const importResultIntoCanvas = useCallback(async (resultId, options = {}) => {
    const shouldNotifyParent = Boolean(options.notifyParent)
    const rid = String(resultId || '').trim()
    if (!rid) return
    if (importingRef.current === rid) return
    importingRef.current = rid
    setStatusMsg(`Importing architecture ${rid}...`)
    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 15000)
      const resp = await apiCall(`/api/v1/import/survivors/${encodeURIComponent(rid)}`, {
        method: 'POST',
        signal: controller.signal,
      })
      clearTimeout(timeout)
      const data = await resp.json()
      if (!resp.ok) {
        throw new Error(formatImportError(data))
      }
      const wf = data.workflow || data
      if (!wf || (wf.schema_version !== 'workflow_graph.v1' && !Array.isArray(wf.nodes))) {
        throw new Error('Import returned unexpected format')
      }
      loadWorkflowJsonRef.current?.(wf)
      importingRef.current = null
      setStatusMsg(`Loaded architecture: ${wf.name || rid}`)
      if (shouldNotifyParent && window.parent && window.parent !== window) {
        window.parent.postMessage({
          source: 'aria_designer', type: 'graph-loaded', resultId: rid,
          name: wf.name, nodeCount: (wf.nodes || []).length, edgeCount: (wf.edges || []).length,
        }, '*')
      }
    } catch (err) {
      importingRef.current = null
      const message = err?.name === 'AbortError' ? 'Import timed out' : (err?.message || String(err))
      setStatusMsg(`Failed to import ${rid}: ${message}`)
      if (shouldNotifyParent && window.parent && window.parent !== window) {
        window.parent.postMessage({
          source: 'aria_designer', type: 'graph-load-error', resultId: rid,
          error: `Failed to import ${rid}: ${message}`,
        }, '*')
      }
    }
  }, [setStatusMsg, loadWorkflowJsonRef, formatImportError])

  // Autosave / embedded graph-changed notification
  useEffect(() => {
    if (embeddedMode) {
      postToParent('graph-changed', { nodeCount: nodes.length, edgeCount: edges.length })
      return
    }
    const timer = setTimeout(() => {
      localStorage.setItem('aria-workflow-autosave', JSON.stringify(buildWorkflowJson(nodes, edges, workflowMeta)))
    }, 1000)
    return () => clearTimeout(timer)
  }, [nodes, edges, workflowMeta, embeddedMode, postToParent])

  // Embedded-ready with retry
  const [readOnly] = useState(() => new URLSearchParams(window.location.search).get('readonly') === '1')
  const loadResultReceived = useRef(false)
  useEffect(() => {
    if (!embeddedMode) return
    loadResultReceived.current = false
    postToParent('embedded-ready', { readOnly: Boolean(readOnly) })
    const retryInterval = setInterval(() => {
      if (loadResultReceived.current) { clearInterval(retryInterval); return }
      postToParent('embedded-ready', { readOnly: Boolean(readOnly) })
    }, 1000)
    return () => clearInterval(retryInterval)
  }, [embeddedMode, postToParent, readOnly])

  // Parent message listener
  const nodesRef = useRef(nodes)
  const edgesRef = useRef(edges)
  nodesRef.current = nodes
  edgesRef.current = edges
  const embeddedModeRef = useRef(embeddedMode)
  embeddedModeRef.current = embeddedMode

  useEffect(() => {
    const handler = (e) => {
      if (e.data?.target !== 'aria_designer') return
      if (e.data.type === 'set-embedded') { setEmbeddedMode(Boolean(e.data.embedded)); return }
      if (!embeddedModeRef.current) return
      if (e.data.type === 'get-graph') {
        postToParent('graph-data', {
          workflow: buildWorkflowJson(nodesRef.current, edgesRef.current, workflowMeta),
          reason: e.data.reason || null, requestId: e.data.requestId || null,
        })
      }
      if (e.data.type === 'load-result' && e.data.resultId) {
        loadResultReceived.current = true
        importResultIntoCanvas(e.data.resultId, { notifyParent: true })
      }
      if (e.data.type === 'load-graph' && e.data.graphJson) {
        try {
          const gj = typeof e.data.graphJson === 'string' ? JSON.parse(e.data.graphJson) : e.data.graphJson
          if (gj.schema_version === 'workflow_graph.v1' || gj.nodes) {
            loadWorkflowJsonRef.current?.(gj)
            setStatusMsg('Loaded morph candidate')
            postToParent('graph-loaded', { name: gj.name || 'morph candidate', nodeCount: (gj.nodes || []).length, edgeCount: (gj.edges || []).length })
          }
        } catch (err) { setStatusMsg(`Failed to load morph candidate: ${err.message}`) }
      }
    }
    window.addEventListener('message', handler)
    return () => window.removeEventListener('message', handler)
  }, [importResultIntoCanvas, postToParent, workflowMeta, loadWorkflowJsonRef, setStatusMsg])

  return {
    embeddedMode, readOnly,
    importResultIntoCanvas,
    postToParent,
  }
}
