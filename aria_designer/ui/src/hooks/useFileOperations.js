import { useCallback, useEffect, useRef, useState } from 'react'
import { apiCall } from '../services/apiService'
import { buildWorkflowJson } from '../utils/workflow'
import { normalizeNodePlacement } from '../utils/layout'

/**
 * File I/O operations: load, save JSON, import, export, clear.
 * @param {Object} deps - { nodes, edges, setNodes, setEdges, components, setComponents,
 *   workflowMeta, setWorkflowMeta, setSelectedNodeId, setStatusMsg, setWorkflowStage,
 *   setRunStatus, fitView, setSaveState, nodeIdCounterRef }
 */
export function useFileOperations({
  nodes, edges, setNodes, setEdges,
  components, setComponents,
  workflowMeta, setWorkflowMeta,
  setSelectedNodeId, setStatusMsg, setWorkflowStage, setRunStatus,
  fitView, setSaveState, getNodeIdCounter, setNodeIdCounter,
}) {
  const loadWorkflowJsonRef = useRef(null)

  const loadWorkflowJson = useCallback((workflow) => {
    if (!workflow || workflow.schema_version !== 'workflow_graph.v1') {
      setStatusMsg('Invalid workflow file')
      return
    }

    const compIndex = new Map(components.map((c) => [c.id, c]))

    // Build port maps from edges for handle creation when registry unavailable
    const targetPorts = {}
    const sourcePorts = {}
    for (const e of workflow.edges || []) {
      const sp = e.source_port || 'y'
      const tp = e.target_port || 'x'
      if (!sourcePorts[e.source]) sourcePorts[e.source] = new Set()
      sourcePorts[e.source].add(sp)
      if (!targetPorts[e.target]) targetPorts[e.target] = new Set()
      targetPorts[e.target].add(tp)
    }

    const nextNodes = (workflow.nodes || []).map((n) => {
      const compType = n.component_type || 'unknown'
      const compId = compType.split('/').pop()
      const category = compType.includes('/') ? compType.split('/')[0] : 'unknown'
      const comp = compIndex.get(compId)

      let label = comp?.name || compId
      if (label === 'unknown' && n.id) {
        label = n.id.replace(/^node_/, '').replace(/_/g, ' ')
      }

      const inputPortNames = targetPorts[n.id] || new Set()
      if (inputPortNames.size === 0 && compId !== 'input' && compId !== 'graph_input') inputPortNames.add('x')
      const outputPortNames = sourcePorts[n.id] || new Set()
      if (outputPortNames.size === 0 && compId !== 'output_head' && compId !== 'graph_output') outputPortNames.add('y')
      const fallbackInputs = [...inputPortNames].map((name) => ({ name, dtype: 'tensor' }))
      const fallbackOutputs = [...outputPortNames].map((name) => ({ name, dtype: 'tensor' }))

      return {
        id: n.id,
        type: 'designer',
        position: n.ui_meta?.position || (n.ui_meta?.x != null ? { x: n.ui_meta.x, y: n.ui_meta.y } : { x: 200, y: 100 }),
        data: {
          label,
          category: comp?.category || category,
          componentId: compType,
          description: comp?.description || '',
          inputs: comp?.inputs || fallbackInputs,
          outputs: comp?.outputs || fallbackOutputs,
          params: comp?.params || {},
          paramValues: n.params || {},
          paramErrors: {},
          performance: comp?.performance || {},
          manifest: comp || {},
          help_md: comp?.help_md || '',
        },
      }
    })

    const nodePortIndex = {}
    for (const n of nextNodes) {
      nodePortIndex[n.id] = {
        inputs: (n.data?.inputs || []).map((p) => p.name),
        outputs: (n.data?.outputs || []).map((p) => p.name),
      }
    }
    const resolvePort = (nodeId, requested, kind) => {
      const ports = kind === 'source'
        ? (nodePortIndex[nodeId]?.outputs || [])
        : (nodePortIndex[nodeId]?.inputs || [])
      if (requested && ports.includes(requested)) return requested
      if (ports.length > 0) return ports[0]
      if (requested) return requested
      return kind === 'source' ? 'y' : 'x'
    }

    const nextEdges = (workflow.edges || []).map((e, idx) => ({
      id: e.id || `e_${idx}`,
      source: e.source,
      target: e.target,
      sourceHandle: resolvePort(e.source, e.source_port, 'source'),
      targetHandle: resolvePort(e.target, e.target_port, 'target'),
    }))

    const currentCounter = getNodeIdCounter()
    const maxId = nextNodes.reduce((max, n) => {
      const m = n.id.match(/(?:^n_|^node_)(\d+)$/)
      if (!m) return max
      return Math.max(max, Number(m[1]))
    }, currentCounter)
    setNodeIdCounter(Math.max(currentCounter, maxId))

    setWorkflowMeta({
      workflow_id: workflow.workflow_id || null,
      name: workflow.name || null,
      metadata: workflow.metadata || {},
    })

    setNodes(normalizeNodePlacement(nextNodes, { grid: [15, 15] }))
    setEdges([])
    setSelectedNodeId(nextNodes[0]?.id || null)
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        setEdges(nextEdges)
        setStatusMsg(`Loaded workflow: ${workflow.name || workflow.workflow_id}`)
        requestAnimationFrame(() => {
          fitView({ padding: 0.1, duration: 200 })
        })
      })
    })
  }, [components, setEdges, setNodes, fitView, setStatusMsg, setSelectedNodeId, setWorkflowMeta, getNodeIdCounter, setNodeIdCounter])

  loadWorkflowJsonRef.current = loadWorkflowJson

  const handleImportFile = useCallback((e) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      try {
        const data = JSON.parse(reader.result)
        loadWorkflowJson(data)
      } catch {
        setStatusMsg('Failed to parse JSON')
      }
    }
    reader.readAsText(file)
    e.target.value = ''
  }, [loadWorkflowJson, setStatusMsg])

  const handleLoadExample = useCallback(async (path) => {
    if (!path) return
    try {
      const normalized = String(path).replace(/^\/+/, '')
      const exampleUrl = new URL(normalized, window.location.href).toString()
      const res = await fetch(exampleUrl)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      loadWorkflowJson(data)
    } catch (err) {
      console.error('Failed to load example', err)
      setStatusMsg(`Failed to load example: ${err.message}`)
    }
  }, [loadWorkflowJson, setStatusMsg])

  useEffect(() => {
    const handler = (e) => {
      if (e.detail) handleLoadExample(e.detail)
    }
    window.addEventListener('load-example', handler)
    return () => window.removeEventListener('load-example', handler)
  }, [handleLoadExample])

  const handleExportJson = useCallback(() => {
    const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
    const blob = new Blob([JSON.stringify(workflow, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `workflow_${workflow.workflow_id}.json`
    a.click()
    URL.revokeObjectURL(url)
    setStatusMsg('Exported JSON')
  }, [nodes, edges, workflowMeta, setStatusMsg])

  const handleExportPython = useCallback(async () => {
    const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
    setStatusMsg('Exporting Python...')
    try {
      const res = await apiCall(`/api/v1/export/onnx`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(workflow),
      })
      const data = await res.json()
      if (!data.success || !data.code) {
        throw new Error(data.error || 'Export python failed')
      }
      const blob = new Blob([data.code], { type: 'text/x-python' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${workflow.workflow_id}.py`
      a.click()
      URL.revokeObjectURL(url)
      setStatusMsg('Exported Python module')
    } catch {
      setStatusMsg('Export Python failed')
    }
  }, [nodes, edges, workflowMeta, setStatusMsg])

  const handleReloadComponents = useCallback(async () => {
    setStatusMsg('Reloading components...')
    try {
      const res = await apiCall(`/api/v1/components/reload`, { method: 'POST' })
      const data = await res.json()
      const list = await apiCall(`/api/v1/components?status=approved`)
      const comps = await list.json()
      setComponents(comps)
      setStatusMsg(`Reloaded components: ${data.reloaded || comps.length}`)
    } catch {
      setStatusMsg('Failed to reload components (API offline)')
    }
  }, [setStatusMsg, setComponents])

  const handleClearCanvas = useCallback(() => {
    if (window.confirm('Are you sure you want to clear the entire canvas? This cannot be undone.')) {
      setNodes([])
      setEdges([])
      setSelectedNodeId(null)
      setWorkflowStage('idle')
      setStatusMsg('Canvas cleared')
      setRunStatus({ phase: 'idle', message: 'Idle', metrics: null })
    }
  }, [setNodes, setEdges, setSelectedNodeId, setWorkflowStage, setStatusMsg, setRunStatus])

  const handleSave = useCallback(async () => {
    const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
    const newId = `wf_${Date.now().toString(36)}`
    const parentId = workflow.workflow_id
    const parentFp = workflowMeta.metadata?.graph_fingerprint || null

    if (parentFp) {
      workflow.metadata = { ...(workflow.metadata || {}), parent_fingerprint: parentFp }
    }
    if (parentId && parentId !== newId) {
      workflow.metadata = { ...(workflow.metadata || {}), parent_workflow_id: parentId }
    }
    workflow.workflow_id = newId

    setSaveState({ phase: 'saving', message: 'Saving\u2026', version: null, fingerprint: null, at: Date.now() })
    setStatusMsg('Saving workflow...')
    try {
      const res = await apiCall(`/api/v1/workflows/${newId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(workflow),
      })
      if (!res.ok) throw new Error(`save ${res.status}`)
      const data = await res.json()
      const fp = data.fingerprint ? ` (fingerprint: ${data.fingerprint.slice(0, 8)}...)` : ''
      const promo = data.auto_promoted && data.promoted_result_id
        ? ` \u2022 promoted ${String(data.promoted_result_id)}`
        : ''
      setWorkflowMeta((prev) => ({
        ...prev,
        workflow_id: newId,
        metadata: {
          ...prev.metadata,
          graph_fingerprint: data.fingerprint || prev.metadata?.graph_fingerprint,
          parent_fingerprint: parentFp,
        },
      }))
      const discoveryUrl = data.fingerprint
        ? `http://localhost:5000/?search=${String(data.fingerprint).slice(0, 12)}`
        : null
      setSaveState({
        phase: 'saved',
        message: `Saved \u00b7 fp ${data.fingerprint ? String(data.fingerprint).slice(0, 12) + '...' : 'n/a'}`,
        version: Number(data.version) || null,
        fingerprint: data.fingerprint || null,
        discoveryUrl,
        at: Date.now(),
      })
      const lineageMsg = parentFp ? ` (parent: ${parentFp.slice(0, 8)}...)` : ''
      setStatusMsg(`Saved as new workflow ${newId}${fp}${lineageMsg}${promo}`)
    } catch (err) {
      const msg = String(err?.message || err || 'save failed')
      const networkish = /failed to fetch|networkerror|network error|load failed/i.test(msg)
      if (networkish) {
        localStorage.setItem('aria-workflow', JSON.stringify(workflow))
        setSaveState({ phase: 'failed', message: 'Save failed (network)', version: null, fingerprint: null, at: Date.now() })
        setStatusMsg('Save failed (network): saved a local browser backup instead')
        return
      }
      setSaveState({ phase: 'failed', message: `Save failed: ${msg}`, version: null, fingerprint: null, at: Date.now() })
      setStatusMsg(`Save failed: ${msg}`)
    }
  }, [nodes, edges, workflowMeta, setStatusMsg, setSaveState, setWorkflowMeta])

  return {
    loadWorkflowJson,
    loadWorkflowJsonRef,
    handleImportFile,
    handleLoadExample,
    handleExportJson,
    handleExportPython,
    handleReloadComponents,
    handleClearCanvas,
    handleSave,
  }
}
