import { apiCall } from "./services/apiService";
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  addEdge,
  Background,
  MarkerType,
  MiniMap,
  ReactFlow,
  useEdgesState,
  useNodesState,
  useReactFlow,
  ReactFlowProvider,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import DesignerNode from './components/DesignerNode'
import GhostNode from './components/GhostNode'
import Palette from './components/Palette'
import Inspector from './components/Inspector'
import PatchPanel from './components/PatchPanel'
import AskAriaModal from './components/AskAriaModal'
import ZoomControls from './components/ZoomControls'
import EmptyState from './components/EmptyState'
import KeyboardShortcuts from './components/KeyboardShortcuts'
import NexusCommandPalette from './components/NexusCommandPalette'
import ImportDialog from './components/ImportDialog'
import RunResultsPanel from './components/RunResultsPanel'
import ErrorBoundary from './components/ErrorBoundary'
import AriaAvatar from './components/AriaAvatar'
import { isValidConnection as validateConnection } from './utils/validation'
import { buildWorkflowJson } from './utils/workflow'
import { findClosestEdge } from './utils/geometry'
import {
  alignNodesHorizontally,
  alignNodesVertically,
  distributeNodesHorizontally,
  distributeNodesVertically,
  findNearestFreePosition,
  getNodeSize,
  normalizeNodePlacement,
  snapPositionToGrid,
  tidySelectedNodes,
} from './utils/layout'
import { starterEdges, starterNodes } from './mockData'

const defaultEdgeOptions = {
  type: 'smoothstep',
  animated: false,
  style: { stroke: '#5a8ab5', strokeWidth: 2 },
  markerEnd: {
    type: MarkerType.ArrowClosed,
    color: '#5a8ab5',
    width: 16,
    height: 16,
  },
}

let nodeIdCounter = 100

function DesignerApp() {
  const [nodes, setNodes, onNodesChange] = useNodesState(starterNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(starterEdges)
  const [selectedNodeId, setSelectedNodeId] = useState(null)
  const [components, setComponents] = useState([])
  const [proposals, setProposals] = useState([])
  const [rightPanelTab, setRightPanelTab] = useState('inspector')
  const [workflowStage, setWorkflowStage] = useState('idle')
  const [previewPatch, setPreviewPatch] = useState(null)
  const [statusMsg, setStatusMsg] = useState('')
  const [runStatus, setRunStatus] = useState({
    phase: 'idle',
    message: 'Idle',
    metrics: null,
  })
  const [showAskAriaModal, setShowAskAriaModal] = useState(false)
  const [showNexusPalette, setShowNexusPalette] = useState(false)
  const [workflowMeta, setWorkflowMeta] = useState({ workflow_id: null, name: null, metadata: {} })
  const [ariaSuggestions, setAriaSuggestions] = useState([])
  const [ariaLoading, setAriaLoading] = useState(false)
  const [saveState, setSaveState] = useState({ phase: 'idle', message: '', version: null, fingerprint: null, at: 0 })

  const [isDragging, setIsDragging] = useState(false)
  const [paletteConstraints, setPaletteConstraints] = useState({})
  const [showShortcuts, setShowShortcuts] = useState(false)
  const [showImportDialog, setShowImportDialog] = useState(false)
  const [arrangeOpen, setArrangeOpen] = useState(false)
  const [fileMenuOpen, setFileMenuOpen] = useState(false)
  const [viewMenuOpen, setViewMenuOpen] = useState(false)
  const [helpRequest, setHelpRequest] = useState(null)
  const [dragGuides, setDragGuides] = useState({ x: null, y: null })
  const [toasts, setToasts] = useState([])
  const [historyUi, setHistoryUi] = useState({ canUndo: false, canRedo: false })
  
  const [snapToGridEnabled, setSnapToGridEnabled] = useState(true)
  const [hardwareView, setHardwareView] = useState(false)
  const [heatmapView, setHeatmapView] = useState(false)
  
  const maxFlops = useMemo(() => {
    let max = 0
    nodes.forEach(n => {
      const f = n.data?.profile?.flops || n.data?.performance?.flops_forward || 0
      if (f > max) max = f
    })
    return max || 1
  }, [nodes])

  const [rightPanelWidth, setRightPanelWidth] = useState(300)
  const [isResizing, setIsResizing] = useState(false)
  const resizeRef = useRef({ startX: 0, startWidth: 300 })

  const startResizing = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    resizeRef.current = { startX: e.clientX, startWidth: rightPanelWidth }
    setIsResizing(true)
  }, [rightPanelWidth])

  const handleResizeKeyDown = useCallback((e) => {
    const step = e.shiftKey ? 40 : 16
    if (e.key === 'ArrowLeft') {
      e.preventDefault()
      setRightPanelWidth((w) => Math.max(250, Math.min(900, w + step)))
      return
    }
    if (e.key === 'ArrowRight') {
      e.preventDefault()
      setRightPanelWidth((w) => Math.max(250, Math.min(900, w - step)))
      return
    }
    if (e.key === 'Home') {
      e.preventDefault()
      setRightPanelWidth(250)
      return
    }
    if (e.key === 'End') {
      e.preventDefault()
      setRightPanelWidth(900)
    }
  }, [])

  useEffect(() => {
    if (!isResizing) return
    const handleMouseMove = (e) => {
      const deltaX = resizeRef.current.startX - e.clientX
      const nextWidth = resizeRef.current.startWidth + deltaX
      setRightPanelWidth(Math.max(250, Math.min(900, nextWidth)))
    }
    const handleMouseUp = () => {
      setIsResizing(false)
    }
    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isResizing])

  // Use a ref to store initial URL params to avoid re-runs
  const initialParams = useMemo(() => new URLSearchParams(window.location.search), [])
  const [readOnly, setReadOnly] = useState(() => initialParams.get('readonly') === '1')
  const [embeddedMode, setEmbeddedMode] = useState(() => initialParams.get('embedded') === '1')
  
  const [evalState, setEvalState] = useState({ stages: [], status: null, totalTimeMs: null, error: null, benchmarking: null })
  const [benchmarkObserved, setBenchmarkObserved] = useState(() => {
    try {
      const raw = localStorage.getItem('aria-benchmark-observed')
      const parsed = raw ? JSON.parse(raw) : {}
      return parsed && typeof parsed === 'object' ? parsed : {}
    } catch {
      return {}
    }
  })
  const [validateUi, setValidateUi] = useState({ inProgress: false, last: 'idle', issues: 0 })
  const [stepStatus, setStepStatus] = useState({
    validate: 'idle',
    compile: 'idle',
    test: 'idle',
    run: 'idle',
  })
  const deepRunAbortRef = useRef(null)
  const validateTimersRef = useRef({})
  const loadWorkflowJsonRef = useRef(null)
  const handleValidateRef = useRef(null)
  const handlePreviewRef = useRef(null)
  const handleSaveRef = useRef(null)
  const handleApplyPatchRef = useRef(null)
  const historyRef = useRef([])
  const futureRef = useRef([])
  const skipHistoryRef = useRef(false)
  const lastSnapshotSigRef = useRef('')
  const reactFlowWrapper = useRef(null)
  const importInputRef = useRef(null)
  const { screenToFlowPosition, deleteElements, fitView, getViewport } = useReactFlow()
  const currentWorkflowId = workflowMeta?.workflow_id || null
  const scopedProposals = useMemo(
    () => (Array.isArray(proposals) ? proposals.filter((p) => p?.workflow_id === currentWorkflowId) : []),
    [proposals, currentWorkflowId]
  )
  const importedBaseline = useMemo(() => {
    const meta = workflowMeta?.metadata || {}
    const resultId = String(meta.result_id || '').trim()
    if (!resultId) return null
    const toNum = (v) => {
      const n = Number(v)
      return Number.isFinite(n) ? n : null
    }
    return {
      resultId,
      lossRatio: toNum(meta.loss_ratio),
      validationLossRatio: toNum(meta.validation_loss_ratio),
      discoveryLossRatio: toNum(meta.discovery_loss_ratio),
      noveltyScore: toNum(meta.novelty_score),
      benchmarkScore: toNum(meta.benchmark_score),
    }
  }, [workflowMeta])
  const proposalQuery = useMemo(() => {
    if (!currentWorkflowId) return null
    const qs = new URLSearchParams({
      status: 'pending',
      workflow_id: String(currentWorkflowId),
      fresh_only: '1',
    })
    return `/api/v1/aria/proposals?${qs.toString()}`
  }, [currentWorkflowId])

  const updateHistoryUi = useCallback(() => {
    setHistoryUi({
      canUndo: historyRef.current.length > 1,
      canRedo: futureRef.current.length > 0,
    })
  }, [])

  const pushToast = useCallback((message, tone = 'info', ttl = 2600) => {
    const id = `t_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`
    setToasts((prev) => [...prev, { id, message, tone }])
    window.setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id))
    }, Math.max(1200, ttl))
  }, [])

  const captureSnapshot = useCallback((nextNodes, nextEdges) => {
    const nodesCopy = JSON.parse(JSON.stringify(nextNodes || []))
    const edgesCopy = JSON.parse(JSON.stringify(nextEdges || []))
    const sig = JSON.stringify({
      n: nodesCopy.map((n) => [n.id, n.position?.x, n.position?.y, n.selected ? 1 : 0]),
      e: edgesCopy.map((e) => [e.id, e.source, e.target, e.sourceHandle || '', e.targetHandle || '', e.selected ? 1 : 0]),
    })
    if (sig === lastSnapshotSigRef.current) return
    historyRef.current.push({ nodes: nodesCopy, edges: edgesCopy, sig })
    if (historyRef.current.length > 120) historyRef.current.shift()
    lastSnapshotSigRef.current = sig
    futureRef.current = []
    updateHistoryUi()
  }, [updateHistoryUi])

  const undoGraph = useCallback(() => {
    if (historyRef.current.length <= 1) return
    const current = historyRef.current.pop()
    if (current) futureRef.current.push(current)
    const prev = historyRef.current[historyRef.current.length - 1]
    if (!prev) return
    skipHistoryRef.current = true
    setNodes(JSON.parse(JSON.stringify(prev.nodes)))
    setEdges(JSON.parse(JSON.stringify(prev.edges)))
    lastSnapshotSigRef.current = prev.sig
    updateHistoryUi()
    pushToast('Undid last graph edit', 'info', 1800)
    window.setTimeout(() => { skipHistoryRef.current = false }, 0)
  }, [setNodes, setEdges, updateHistoryUi, pushToast])

  const redoGraph = useCallback(() => {
    if (futureRef.current.length === 0) return
    const next = futureRef.current.pop()
    if (!next) return
    historyRef.current.push(next)
    skipHistoryRef.current = true
    setNodes(JSON.parse(JSON.stringify(next.nodes)))
    setEdges(JSON.parse(JSON.stringify(next.edges)))
    lastSnapshotSigRef.current = next.sig
    updateHistoryUi()
    pushToast('Redid graph edit', 'info', 1800)
    window.setTimeout(() => { skipHistoryRef.current = false }, 0)
  }, [setNodes, setEdges, updateHistoryUi, pushToast])

  const importingRef = useRef(null)
  const importResultIntoCanvas = useCallback(async (resultId, options = {}) => {
    const shouldNotifyParent = Boolean(options.notifyParent)
    const rid = String(resultId || '').trim()
    if (!rid) return
    // Prevent concurrent/duplicate imports for the same result
    if (importingRef.current === rid) {
      console.log('[Designer] importResultIntoCanvas skipped (already importing)', rid)
      return
    }
    importingRef.current = rid
    console.log('[Designer] importResultIntoCanvas called for', rid, 'notifyParent=', shouldNotifyParent)
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
      const wf = data.workflow || data
      if (!wf || (wf.schema_version !== 'workflow_graph.v1' && !Array.isArray(wf.nodes))) {
        throw new Error('Import returned unexpected format')
      }
      console.log('[Designer] import response for', rid, ':',
        (wf.nodes || []).length, 'nodes,',
        (wf.edges || []).length, 'edges,',
        'schema:', wf.schema_version)
      loadWorkflowJsonRef.current?.(wf)
      importingRef.current = null
      setStatusMsg(`Loaded architecture: ${wf.name || rid}`)
      if (shouldNotifyParent && window.parent && window.parent !== window) {
        console.log('[Designer] posting graph-loaded to parent for', rid, '(',
          (wf.nodes || []).length, 'nodes,', (wf.edges || []).length, 'edges)')
        window.parent.postMessage({
          source: 'aria_designer',
          type: 'graph-loaded',
          resultId: rid,
          name: wf.name,
          nodeCount: (wf.nodes || []).length,
          edgeCount: (wf.edges || []).length,
        }, '*')
      } else {
        console.warn('[Designer] NOT posting graph-loaded:',
          'notifyParent=', shouldNotifyParent,
          'hasParent=', window.parent !== window)
      }
    } catch (err) {
      importingRef.current = null
      const message = err?.name === 'AbortError'
        ? 'Import timed out'
        : (err?.message || String(err))
      setStatusMsg(`Failed to import ${rid}: ${message}`)
      if (shouldNotifyParent && window.parent && window.parent !== window) {
        window.parent.postMessage({
          source: 'aria_designer',
          type: 'graph-load-error',
          resultId: rid,
          error: `Failed to import ${rid}: ${message}`,
        }, '*')
      }
    }
  }, [])

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

  // Fetch components from API
  useEffect(() => {
    apiCall(`/api/v1/components?status=approved`)
      .then((r) => r.json())
      .then((data) => {
        setComponents(data)
        setStatusMsg(`${data.length} components loaded`)
      })
      .catch(() => {
        setStatusMsg('API offline — using mock palette')
        // Fallback: import mock data
        import('./mockData').then((m) => {
          setComponents(m.palette.map((p) => ({
            id: p.id, name: p.label, category: p.category,
            inputs: [{ name: 'x', dtype: 'tensor' }],
            outputs: [{ name: 'y', dtype: 'tensor' }],
          })))
        })
      })
  }, [])

  // Fetch proposals from API.
  // Poll only when the Proposals panel is active and back off on failures
  // to avoid flooding proxy logs when Aria proposal routes are unavailable.
  useEffect(() => {
    if (!proposalQuery || rightPanelTab !== 'proposals') {
      return undefined
    }
    let cancelled = false
    let timer = null

    const schedule = (ms) => {
      if (cancelled) return
      timer = setTimeout(fetchProposals, ms)
    }

    const fetchProposals = async () => {
      if (cancelled) return
      try {
        const r = await apiCall(proposalQuery)
        const data = await r.json()
        setProposals(Array.isArray(data) ? data : [])
        schedule(5000)
      } catch (err) {
        // Keep UI resilient if designer API proxy is temporarily unavailable.
        setProposals([])
        console.warn('Failed to fetch proposals; backing off poll', err)
        schedule(30000)
      }
    }

    fetchProposals()
    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
  }, [proposalQuery, rightPanelTab])

  // Fetch palette constraints
  useEffect(() => {
    const fetchConstraints = async () => {
      try {
        const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
        const res = await apiCall(`/api/v1/constraints/palette`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ workflow, selected_node_id: selectedNodeId }),
        })
        if (res.ok) {
          const data = await res.json()
          setPaletteConstraints(data)
        }
      } catch (err) {
        console.warn('Failed to fetch palette constraints', err)
      }
    }
    const timer = setTimeout(fetchConstraints, 300)
    return () => clearTimeout(timer)
  }, [nodes, edges, workflowMeta, selectedNodeId])

  // PostMessage bridge — notify parent window of state changes in embedded mode.
  const postToParent = useCallback((type, payload = {}) => {
    if (!embeddedMode) return
    if (!window.parent || window.parent === window) {
      if (type === 'embedded-ready' || type === 'graph-loaded' || type === 'graph-load-error') {
        console.warn('[Designer] postToParent BLOCKED: no parent window or is top', type)
      }
      return
    }
    console.log('[Designer] postToParent:', type, payload)
    window.parent.postMessage({ source: 'aria_designer', type, ...payload }, '*')
  }, [embeddedMode])

  // Autosave local draft on canvas changes (skip in embedded mode).
  useEffect(() => {
    if (embeddedMode) {
      // Notify parent of graph changes
      postToParent('graph-changed', { nodeCount: nodes.length, edgeCount: edges.length })
      return
    }
    const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
    localStorage.setItem('aria-workflow-autosave', JSON.stringify(workflow))
  }, [nodes, edges, workflowMeta, embeddedMode, postToParent])

  useEffect(() => {
    try {
      localStorage.setItem('aria-benchmark-observed', JSON.stringify(benchmarkObserved || {}))
    } catch {
      // Ignore storage write failures.
    }
  }, [benchmarkObserved])

  // Keep refs for nodes/edges so the message handler can access current
  // values without causing the effect to re-run on every graph change.
  const nodesRef = useRef(nodes)
  const edgesRef = useRef(edges)
  useEffect(() => { nodesRef.current = nodes }, [nodes])
  useEffect(() => { edgesRef.current = edges }, [edges])

  useEffect(() => {
    if (historyRef.current.length > 0) return
    captureSnapshot(nodes, edges)
  }, [captureSnapshot, nodes, edges])

  useEffect(() => {
    if (skipHistoryRef.current) return
    const timer = setTimeout(() => {
      if (skipHistoryRef.current) return
      captureSnapshot(nodes, edges)
    }, 220)
    return () => clearTimeout(timer)
  }, [nodes, edges, captureSnapshot])

  // Post embedded-ready when entering embedded mode, with retries until parent
  // sends load-result (indicating it received our ready signal).
  const loadResultReceived = useRef(false)
  useEffect(() => {
    if (!embeddedMode) return
    loadResultReceived.current = false
    console.log('[Designer] posting embedded-ready to parent')
    postToParent('embedded-ready', { readOnly: Boolean(readOnly) })
    const retryInterval = setInterval(() => {
      if (loadResultReceived.current) { clearInterval(retryInterval); return }
      console.log('[Designer] retrying embedded-ready')
      postToParent('embedded-ready', { readOnly: Boolean(readOnly) })
    }, 1000)
    return () => clearInterval(retryInterval)
  }, [embeddedMode, postToParent, readOnly])

  // Listen for commands from parent window (e.g., request current graph).
  const embeddedModeRef = useRef(embeddedMode)
  embeddedModeRef.current = embeddedMode
  useEffect(() => {
    // Always listen — parent may toggle embedded mode via set-embedded.
    const handler = (e) => {
      if (e.data?.target !== 'aria_designer') return
      // set-embedded is always handled, regardless of current mode
      if (e.data.type === 'set-embedded') {
        setEmbeddedMode(Boolean(e.data.embedded))
        return
      }
      if (!embeddedModeRef.current) return
      if (e.data.type === 'get-graph') {
        const workflow = buildWorkflowJson(nodesRef.current, edgesRef.current, workflowMeta)
        postToParent('graph-data', {
          workflow,
          reason: e.data.reason || null,
          requestId: e.data.requestId || null,
        })
      }
      if (e.data.type === 'load-result' && e.data.resultId) {
        loadResultReceived.current = true
        console.log('[Designer] received load-result for', e.data.resultId)
        importResultIntoCanvas(e.data.resultId, { notifyParent: true })
      }
      if (e.data.type === 'load-graph' && e.data.graphJson) {
        try {
          const gj = typeof e.data.graphJson === 'string' ? JSON.parse(e.data.graphJson) : e.data.graphJson
          if (gj.schema_version === 'workflow_graph.v1' || gj.nodes) {
            loadWorkflowJsonRef.current?.(gj)
            setStatusMsg('Loaded morph candidate')
            postToParent('graph-loaded', {
              name: gj.name || 'morph candidate',
              nodeCount: (gj.nodes || []).length,
              edgeCount: (gj.edges || []).length,
            })
          }
        } catch (err) {
          setStatusMsg(`Failed to load morph candidate: ${err.message}`)
        }
      }
    }
    window.addEventListener('message', handler)
    return () => window.removeEventListener('message', handler)
  }, [importResultIntoCanvas, postToParent])

  // URL param handling — load a workflow from research pipeline when embedded.
  // Supports: ?import_result_id=res_xxx
  const urlParamsHandled = useRef(false)
  useEffect(() => {
    if (urlParamsHandled.current || components.length === 0) return
    const resultId = initialParams.get('import_result_id')

    if (resultId) {
      urlParamsHandled.current = true
      importResultIntoCanvas(resultId, { notifyParent: embeddedMode })
    }
  }, [components, importResultIntoCanvas, embeddedMode])

  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault()
        setShowNexusPalette(prev => !prev)
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleNexusAction = useCallback((action) => {
    if (action.id === 'nav-dashboard') {
      window.location.href = '/research/dashboard'
    } else if (action.id === 'action-validate') {
      handleValidateRef.current?.()
    } else if (action.id === 'action-run') {
      handlePreviewRef.current?.()
    } else if (action.id === 'action-save') {
      handleSaveRef.current?.()
    } else if (action.id.startsWith('add-node-')) {
      const c = action.payload
      const newNode = {
        id: `node_${Date.now()}`,
        type: 'designer',
        position: { x: 400, y: 300 },
        data: {
          label: c.name,
          category: c.category,
          component_type: c.id,
          inputs: c.inputs || [],
          outputs: c.outputs || [],
          params: {},
        },
      }
      setNodes((nds) => [...nds, newNode])
      setStatusMsg(`Added ${c.name}`)
    }
  }, [setNodes])

  useEffect(() => () => {
    Object.values(validateTimersRef.current).forEach((t) => clearTimeout(t))
  }, [])

  const selectedNode = useMemo(
    () => nodes.find((n) => n.id === selectedNodeId) || null,
    [nodes, selectedNodeId]
  )

  const openNodeHelp = useCallback((nodeId) => {
    setSelectedNodeId(nodeId)
    setRightPanelTab('inspector')
    setHelpRequest({ nodeId, ts: Date.now() })
  }, [])

  const nodeTypes = useMemo(
    () => ({
      designer: (nodeProps) => (
        <DesignerNode
          {...nodeProps}
          onHelp={() => openNodeHelp(nodeProps.id)}
          hardwareView={hardwareView}
          heatmapView={heatmapView}
          maxFlops={maxFlops}
        />
      ),
      ghost: GhostNode,
    }),
    [openNodeHelp, hardwareView, heatmapView, maxFlops]
  )

  const exampleOptions = useMemo(() => ([
    { label: 'Simple Linear', value: '/examples/simple_linear.json' },
    { label: 'Tropical Attention', value: '/examples/tropical_attention.json' },
    { label: 'Tropical Block', value: '/examples/tropical_block.json' },
    { label: 'Transformer Mini', value: '/examples/transformer_mini.json' },
    { label: 'SSM Stack', value: '/examples/ssm_stack.json' },
    { label: 'Hybrid Attn+SSM+MoE', value: '/examples/hybrid_attn_ssm_moe.json' },
    { label: 'Adaptive Tri-Lane v1', value: '/examples/adaptive_trilane_v1.json' },
    { label: 'Adaptive Tri-Lane v2 (Residual+SSM)', value: '/examples/adaptive_trilane_v2.json' },
    { label: 'Adaptive Tri-Lane v3 (Tropical+Hyp)', value: '/examples/adaptive_trilane_v3.json' },
  ]), [])

  const handleGhostClick = useCallback((suggestion) => {
    const c = suggestion.component
    const componentType = c.id.includes('/') ? c.id : `${c.category || 'math'}/${c.id}`
    
    // Find a good position (usually where the ghost was)
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
    
    // Auto-apply logic similar to handleAskAriaSubmit
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
  }, [nodes, workflowMeta, pushToast])

  const displayNodes = nodes

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge({ ...params }, eds)),
    [setEdges]
  )

  const onNodeDragStop = useCallback(
    (_, node) => {
      const desiredPosition = snapToGridEnabled
        ? snapPositionToGrid(node.position, [15, 15])
        : node.position
      setNodes((nds) => {
        const resolved = findNearestFreePosition(node.id, desiredPosition, nds, { grid: [15, 15] })
        return nds.map((n) => (n.id === node.id ? { ...n, position: resolved } : n))
      })

      // Auto-injection for existing nodes
      const { edge, distance } = findClosestEdge(desiredPosition, edges, nodes)
      
      // Ensure we don't try to inject into an edge that is already connected to this node
      if (edge && distance < 25 && edge.source !== node.id && edge.target !== node.id) {
        const c1 = {
          source: edge.source,
          sourceHandle: edge.sourceHandle,
          target: node.id,
          targetHandle: node.data.inputs[0]?.name || 'x',
        }
        const c2 = {
          source: node.id,
          sourceHandle: node.data.outputs[0]?.name || 'y',
          target: edge.target,
          targetHandle: edge.targetHandle,
        }
        const filteredPreview = edges.filter((e) => e.id !== edge.id)
        const canInject = validateConnection(c1, nodes, filteredPreview)
          && validateConnection(c2, nodes, [...filteredPreview, { id: 'tmp_inject_1', ...c1 }])
        if (!canInject) {
          setStatusMsg('Skipped auto-insert: incompatible ports or invalid edge direction')
          pushToast('Skipped auto-insert: incompatible edge rewrite', 'warn')
          setDragGuides({ x: null, y: null })
          return
        }
        setEdges((eds) => {
          const filtered = eds.filter((e) => e.id !== edge.id)
          return [
            ...filtered,
            { id: `e_inject_1_${Date.now()}`, ...c1 },
            { id: `e_inject_2_${Date.now()}`, ...c2 },
          ]
        })
      }
      setDragGuides({ x: null, y: null })
    },
    [edges, nodes, setEdges, setNodes, snapToGridEnabled, pushToast]
  )

  const onNodeDrag = useCallback((_, node) => {
    const viewport = getViewport()
    const zoom = Number(viewport?.zoom) || 1
    const threshold = 8 / zoom
    const draggedSize = getNodeSize(node)
    const dragX = node.position.x
    const dragY = node.position.y
    const dragCx = dragX + draggedSize.width / 2
    const dragCy = dragY + draggedSize.height / 2
    const dragRight = dragX + draggedSize.width
    const dragBottom = dragY + draggedSize.height

    let bestX = { dist: Infinity, value: null, offset: 0 }
    let bestY = { dist: Infinity, value: null, offset: 0 }
    const xAnchors = [
      { value: dragX, offset: 0 },
      { value: dragCx, offset: draggedSize.width / 2 },
      { value: dragRight, offset: draggedSize.width },
    ]
    const yAnchors = [
      { value: dragY, offset: 0 },
      { value: dragCy, offset: draggedSize.height / 2 },
      { value: dragBottom, offset: draggedSize.height },
    ]

    for (const other of nodes) {
      if (other.id === node.id) continue
      const sz = getNodeSize(other)
      const ox = other.position.x
      const oy = other.position.y
      const oxc = ox + sz.width / 2
      const oyc = oy + sz.height / 2
      const oright = ox + sz.width
      const obottom = oy + sz.height

      const xCandidates = [ox, oxc, oright]
      const yCandidates = [oy, oyc, obottom]
      for (const dx of xAnchors) {
        for (const tx of xCandidates) {
          const dist = Math.abs(dx.value - tx)
          if (dist <= threshold && dist < bestX.dist) bestX = { dist, value: tx, offset: dx.offset }
        }
      }
      for (const dy of yAnchors) {
        for (const ty of yCandidates) {
          const dist = Math.abs(dy.value - ty)
          if (dist <= threshold && dist < bestY.dist) bestY = { dist, value: ty, offset: dy.offset }
        }
      }
    }

    const snappedX = Number.isFinite(bestX.value) ? (bestX.value - bestX.offset) : dragX
    const snappedY = Number.isFinite(bestY.value) ? (bestY.value - bestY.offset) : dragY
    setDragGuides({
      x: Number.isFinite(bestX.value) ? bestX.value : null,
      y: Number.isFinite(bestY.value) ? bestY.value : null,
    })
    if (Math.abs(snappedX - dragX) > 0.25 || Math.abs(snappedY - dragY) > 0.25) {
      setNodes((nds) => nds.map((n) => (n.id === node.id ? { ...n, position: { x: snappedX, y: snappedY } } : n)))
    }
  }, [nodes, getViewport])

  // Drag-and-drop from palette onto canvas
  const onDragOver = useCallback((e) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
    setIsDragging(true)
  }, [])

  const onDrop = useCallback(
    (e) => {
      e.preventDefault()
      setIsDragging(false)
      const raw = e.dataTransfer.getData('application/aria-component')
      if (!raw) return

      const comp = JSON.parse(raw)
      const compConstraint = paletteConstraints?.[comp.id]
      if (compConstraint?.compatible === false) {
        const reason = Array.isArray(compConstraint.reasons) && compConstraint.reasons.length > 0
          ? compConstraint.reasons[0]
          : 'Current graph state does not allow this component here'
        setStatusMsg(`Cannot place ${comp.name || comp.id}: ${reason}`)
        pushToast(`Cannot place ${comp.name || comp.id}: ${reason}`, 'warn', 3200)
        return
      }
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY })
      const desiredPos = snapToGridEnabled ? snapPositionToGrid(position, [15, 15]) : position
      const newId = `n_${++nodeIdCounter}`

      const newNode = {
        id: newId,
        type: 'designer',
        position: desiredPos,
        data: {
          label: comp.name || comp.id,
          category: comp.category,
          componentId: comp.id,
          description: comp.description || '',
          inputs: comp.inputs || [],
          outputs: comp.outputs || [],
          params: comp.params || {},
          paramValues: {},
          paramErrors: {},
          performance: comp.performance || {},
          manifest: comp,
        },
      }
      setNodes((prev) => normalizeNodePlacement([...prev, newNode], { grid: [15, 15] }))
      setSelectedNodeId(newId)

      // Auto-injection logic: check if dropped on/near an edge
      const { edge, distance } = findClosestEdge(desiredPos, edges, nodes)
      if (edge && distance < 25) {
        const c1 = {
          source: edge.source,
          sourceHandle: edge.sourceHandle,
          target: newId,
          targetHandle: newNode.data.inputs[0]?.name || 'x',
        }
        const c2 = {
          source: newId,
          sourceHandle: newNode.data.outputs[0]?.name || 'y',
          target: edge.target,
          targetHandle: edge.targetHandle,
        }
        const nextNodes = [...nodes, newNode]
        const filteredPreview = edges.filter((e) => e.id !== edge.id)
        const canInject = validateConnection(c1, nextNodes, filteredPreview)
          && validateConnection(c2, nextNodes, [...filteredPreview, { id: 'tmp_drop_inject_1', ...c1 }])
        if (!canInject) {
          setStatusMsg(`Added ${comp.name || comp.id}; skipped auto-connect (incompatible edge insertion)`)
          pushToast(`Added ${comp.name || comp.id}; auto-connect skipped`, 'info')
          return
        }
        setEdges((eds) => {
          const filtered = eds.filter((e) => e.id !== edge.id)
          return [
            ...filtered,
            { id: `e_inject_1_${Date.now()}`, ...c1 },
            { id: `e_inject_2_${Date.now()}`, ...c2 },
          ]
        })
      }
    },
    [screenToFlowPosition, setNodes, edges, nodes, setEdges, snapToGridEnabled, paletteConstraints, pushToast]
  )

  const onAddFromPalette = useCallback(
    (comp) => {
      const newId = `n_${++nodeIdCounter}`
      const newNode = {
        id: newId,
        type: 'designer',
        position: snapPositionToGrid({ x: 200 + (nodeIdCounter % 5) * 50, y: 100 + (nodeIdCounter % 7) * 60 }, [15, 15]),
        data: {
          label: comp.name || comp.id,
          category: comp.category,
          componentId: comp.id,
          description: comp.description || '',
          inputs: comp.inputs || [],
          outputs: comp.outputs || [],
          params: comp.params || {},
          paramValues: {},
          paramErrors: {},
          performance: comp.performance || {},
          manifest: comp,
        },
      }
      setNodes((prev) => normalizeNodePlacement([...prev, newNode], { grid: [15, 15] }))
      setSelectedNodeId(newId)
    },
    [setNodes]
  )

  const validateNodeConfig = useCallback((nodeId, componentId, config) => {
    if (!componentId) return
    const timerKey = String(nodeId)
    if (validateTimersRef.current[timerKey]) {
      clearTimeout(validateTimersRef.current[timerKey])
    }
    validateTimersRef.current[timerKey] = setTimeout(async () => {
      try {
        const res = await apiCall(`/api/v1/components/${componentId}/validate-config`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ config }),
        })
        const data = await res.json()
        const paramErrors = {}
        for (const err of data?.errors || []) {
          if (!err?.param) continue
          if (!paramErrors[err.param]) paramErrors[err.param] = []
          paramErrors[err.param].push(err.message || 'Invalid value')
        }
        setNodes((prev) =>
          prev.map((n) =>
            n.id === nodeId
              ? { ...n, data: { ...n.data, paramErrors } }
              : n
          )
        )
      } catch {
        // API offline: keep local editor usable.
      }
    }, 250)
  }, [setNodes])

  // Update node param
  const onParamChange = useCallback(
    (nodeId, paramName, value) => {
      let nextComponentId = ''
      let nextParamValues = {}
      setNodes((prev) =>
        prev.map((n) =>
          n.id === nodeId
            ? (() => {
                nextParamValues = { ...n.data.paramValues, [paramName]: value }
                nextComponentId = String(n.data?.componentId || '').split('/').pop()
                return { ...n, data: { ...n.data, paramValues: nextParamValues } }
              })()
            : n
        )
      )
      if (nextComponentId) {
        validateNodeConfig(nodeId, nextComponentId, nextParamValues)
      }
    },
    [setNodes, validateNodeConfig]
  )

  const [isRunDisabled, setIsRunDisabled] = useState(false)
  
  // Real-time connection validation (prevents cycles, duplicates, port conflicts)
  const isValidConnection = useCallback((connection) => {
    return validateConnection(connection, nodes, edges)
  }, [nodes, edges, workflowMeta])

  // Live graph validation effect
  useEffect(() => {
    const timer = setTimeout(() => {
      const newErrors = {}
      let hasBlockingErrors = false

      // Check 1: Required parameters
      nodes.forEach(n => {
        const manifest = n.data.manifest
        if (!manifest) return
        const nodeErrors = []
        
        // Simple check for missing params that have no default
        // (This is a heuristic as schema doesn't explicitly mark required yet)
        
        if (nodeErrors.length > 0) {
          newErrors[n.id] = nodeErrors
          hasBlockingErrors = true
        }
      })

      // Check 2: Disconnected inputs (except graph inputs)
      const targetIds = new Set(edges.map(e => e.target))
      nodes.forEach(n => {
        if (n.data.componentId === 'io/input' || n.data.componentId === 'io/graph_input') return
        // If node requires inputs but has none
        const requiredInputs = n.data.manifest?.inputs?.length || 0
        if (requiredInputs > 0 && !targetIds.has(n.id)) {
           // We won't block run, but we warn
           // hasBlockingErrors = true 
        }
      })

      // Update UI if errors changed
      // We need to be careful not to cause infinite loops. 
      // Only update if current data.errors differs from newErrors.
      let changed = false
      const nextNodes = nodes.map(n => {
        const currentErr = n.data.errors || []
        const nextErr = newErrors[n.id] || []
        if (JSON.stringify(currentErr) !== JSON.stringify(nextErr)) {
          changed = true
          return {
            ...n,
            className: nextErr.length > 0 ? 'node-invalid' : '',
            data: { ...n.data, errors: nextErr }
          }
        }
        return n
      })

      if (changed) {
        setNodes(nextNodes)
      }
      setIsRunDisabled(hasBlockingErrors)

    }, 500)
    return () => clearTimeout(timer)
  }, [nodes, edges, setNodes])

  // Actions
  const handleValidate = useCallback(async () => {
    const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
    setWorkflowStage('validate')
    setStepStatus((s) => ({ ...s, validate: 'running' }))
    setValidateUi({ inProgress: true, last: 'idle', issues: 0 })
    setStatusMsg('Validating workflow...')
    setRunStatus({ phase: 'running', message: 'Validation in progress...', metrics: null })
    clearNodeHighlights()
    try {
      let data
      const res = await apiCall(`/api/v1/workflows/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflow }),
      })
      if (!res.ok) {
        // Only throw if validation hard-failed to return details
        if (res.status !== 400 && res.status !== 422) {
           throw new Error(`validate ${res.status}`)
        }
      }
      data = await res.json()

      // Treat response with no errors as valid even if valid field is missing
      const hasErrors = (data.issues || []).some((i) => i.severity === 'error')
      if (data.success === true || data.valid === true || (data.issues && !hasErrors)) {
        setStepStatus((s) => ({ ...s, validate: 'pass' }))
        setValidateUi({ inProgress: false, last: 'pass', issues: 0 })
        setStatusMsg('Validation passed: no issues found')
        setRunStatus({ phase: 'success', message: 'Validation passed', metrics: { issues: 0 } })
        return true
      }

      const errorMap = {}
      if (data.node_statuses) {
        for (const [nodeId, info] of Object.entries(data.node_statuses)) {
          if (!info.valid) errorMap[nodeId] = info.errors || ['Invalid node']
        }
      }
      if (data.issues) {
        for (const issue of data.issues) {
          if (issue.node_id) {
            if (!errorMap[issue.node_id]) errorMap[issue.node_id] = []
            errorMap[issue.node_id].push(issue.message || 'Validation issue')
          }
        }
      }
      highlightNodeErrors(errorMap)

      const globalErrors = data.global_errors || []
      const issueErrors = (data.issues || []).map((i) => i.message).filter(Boolean)
      const messages = [...globalErrors, ...issueErrors]
      const issueCount = Object.values(errorMap).reduce((acc, arr) => acc + (arr?.length || 0), 0) || messages.length || 1
      setStepStatus((s) => ({ ...s, validate: 'fail' }))
      setValidateUi({ inProgress: false, last: 'fail', issues: issueCount })
      setStatusMsg(messages.length > 0 ? `Validation failed: ${messages.join('; ')}` : 'Validation failed')
      setRunStatus({ phase: 'failed', message: 'Validation failed', metrics: { issues: issueCount } })
    } catch {
      setStepStatus((s) => ({ ...s, validate: 'fail' }))
      setValidateUi({ inProgress: false, last: 'fail', issues: 1 })
      setStatusMsg('Validation failed (API offline)')
      setRunStatus({ phase: 'failed', message: 'Validation failed (API offline)', metrics: null })
    }
  }, [nodes, edges, clearNodeHighlights, highlightNodeErrors])

  const handleCompile = useCallback(async () => {
    // Auto-validate first if not already passed
    if (stepStatus.validate !== 'pass') {
      const passed = await handleValidate()
      if (!passed) {
        setStatusMsg('Compile skipped: fix validation errors first')
        return false
      }
    }
    const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
    setWorkflowStage('compile')
    setStepStatus((s) => ({ ...s, compile: 'running' }))
    clearNodeHighlights()
    setStatusMsg('Compiling workflow...')
    setRunStatus({ phase: 'compiling', message: 'Compiling workflow...', metrics: null })
    try {
      let data
      let usedDesignerApi = true
      const res = await apiCall(`/api/v1/workflows/compile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflow, target: 'cpu' }),
      })
      if (!res.ok) throw new Error(`compile ${res.status}`)
      data = await res.json()

      const compiled = data.success === true || data.compiled === true
      if (compiled) {
        setStepStatus((s) => ({ ...s, compile: 'pass' }))
        if (usedDesignerApi) {
          setStatusMsg(`Compiled successfully: ${data.n_ops || 0} ops, ${data.param_count || 0} params`)
          setRunStatus({
            phase: 'success',
            message: 'Compile succeeded',
            metrics: {
              params: data.param_count || 0,
              ops: data.n_ops || 0,
              depth: data.depth || 0,
            },
          })
        } else {
          setStatusMsg(`Compiled successfully: ${data.submodule_count || 0} submodules.`)
          setRunStatus({
            phase: 'success',
            message: 'Compile succeeded',
            metrics: {
              submodules: data.submodule_count || 0,
            },
          })
        }
        return true
      } else {
        setStepStatus((s) => ({ ...s, compile: 'fail' }))
        const err = data.error || 'Compilation failed'
        const errorMap = collectFailureErrorMap(data, err)
        if (Object.keys(errorMap).length > 0) {
          highlightNodeErrors(errorMap)
        }
        const warning = Array.isArray(data.semantic_warnings) && data.semantic_warnings.length > 0
          ? String(data.semantic_warnings[0].message || data.semantic_warnings[0])
          : null
        const missingKernel = String(err).includes('Missing runtime kernel_fallback.py')
        const guidance = missingKernel ? 'Missing component runtime kernel(s). Replace that node or add kernel_fallback.' : null
        setStatusMsg(
          `Compilation failed: ${err}` +
          `${guidance ? ` — ${guidance}` : ''}` +
          `${warning ? ` — warning: ${warning}` : ''}`
        )
        setRunStatus({
          phase: 'failed',
          message: `Compile failed: ${err}`,
          metrics: {
            ...(guidance ? { hint: guidance } : {}),
            ...(warning ? { warning } : {}),
          },
        })
        return false
      }
    } catch {
      setStepStatus((s) => ({ ...s, compile: 'fail' }))
      setStatusMsg('Compilation failed (API offline)')
      setRunStatus({ phase: 'failed', message: 'Compile failed (API offline)', metrics: null })
      return false
    }
  }, [nodes, edges, workflowMeta, stepStatus, handleValidate, clearNodeHighlights, highlightNodeErrors, collectFailureErrorMap])

  const handleSave = useCallback(async () => {
    const workflow = buildWorkflowJson(nodes, edges, workflowMeta)

    // Always save to a new workflow_id so each fingerprint is distinct.
    // This gives us automatic lineage: each save is a new "fork" that
    // tracks its parent fingerprint via metadata.
    const newId = `wf_${Date.now().toString(36)}`
    const parentId = workflow.workflow_id
    const parentFp = workflowMeta.metadata?.graph_fingerprint || null

    // Store parent lineage in metadata so the backend can track it
    if (parentFp) {
      workflow.metadata = { ...(workflow.metadata || {}), parent_fingerprint: parentFp }
    }
    if (parentId && parentId !== newId) {
      workflow.metadata = { ...(workflow.metadata || {}), parent_workflow_id: parentId }
    }
    workflow.workflow_id = newId

    setSaveState({ phase: 'saving', message: 'Saving…', version: null, fingerprint: null, at: Date.now() })
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
        ? ` • promoted ${String(data.promoted_result_id)}`
        : ''
      // Update workflowMeta to the new ID and fingerprint
      setWorkflowMeta((prev) => ({
        ...prev,
        workflow_id: newId,
        metadata: {
          ...prev.metadata,
          graph_fingerprint: data.fingerprint || prev.metadata?.graph_fingerprint,
          parent_fingerprint: parentFp,
        },
      }))
      setSaveState({
        phase: 'saved',
        message: `Saved · fp ${data.fingerprint ? String(data.fingerprint).slice(0, 12) + '...' : 'n/a'}`,
        version: Number(data.version) || null,
        fingerprint: data.fingerprint || null,
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
  }, [nodes, edges, workflowMeta])

  useEffect(() => {
    if (!saveState?.phase || saveState.phase === 'idle' || saveState.phase === 'saving') return
    const t = setTimeout(() => {
      setSaveState((prev) => (prev.phase === 'saving' ? prev : {
        phase: 'idle',
        message: '',
        version: prev.version || null,
        fingerprint: prev.fingerprint || null,
        at: prev.at || 0,
      }))
    }, 5000)
    return () => clearTimeout(t)
  }, [saveState])

  const handlePreview = useCallback(async () => {
    // Auto-validate + compile first
    if (stepStatus.compile !== 'pass') {
      const compiled = await handleCompile()
      if (!compiled) {
        setStatusMsg('Test skipped: fix compile errors first')
        return false
      }
    }
    const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
    try {
      setWorkflowStage('run')
      setStepStatus((s) => ({ ...s, test: 'running' }))
      clearNodeHighlights()
      setStatusMsg('Running preview...')
      setRunStatus({ phase: 'running', message: 'Running forward pass...', metrics: null })
      let data
      let usedDesignerApi = true
      try {
        const res = await apiCall(`/api/v1/workflows/preview`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ workflow }),
        })
        if (!res.ok) throw new Error(`preview ${res.status}`)
        data = await res.json()
      } catch {
        usedDesignerApi = false
        const res = await apiCall(`/api/v1/workflows/preview`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ workflow }),
        })
        data = await res.json()
      }

      if (data.success === true) {
        setStepStatus((s) => ({ ...s, test: 'pass' }))
        if (usedDesignerApi && data.metrics) {
          setStatusMsg(
            `Run complete — params ${data.metrics.param_count || 0}, FLOPs/token ${data.metrics.flops_per_token || 0}, ` +
            `forward ${data.metrics.forward_ms || 0}ms`
          )
          setRunStatus({
            phase: 'success',
            message: 'Run succeeded',
            metrics: {
              params: data.metrics.param_count || 0,
              flops: data.metrics.flops_per_token || 0,
              memoryMb: data.metrics.peak_memory_mb || 0,
              forwardMs: data.metrics.forward_ms || 0,
            },
          })
        } else {
          const nResults = Object.keys(data.results || {}).length
          setStatusMsg(`Run complete — ${nResults} output${nResults !== 1 ? 's' : ''} computed`)
          setRunStatus({
            phase: 'success',
            message: 'Run succeeded',
            metrics: { outputs: nResults },
          })
          setNodes(nds => nds.map(n => {
            if (data.results[n.id]) {
              return {
                ...n,
                data: { ...n.data, preview: data.results[n.id] }
              }
            }
            return n
          }))
        }
      } else {
        setStepStatus((s) => ({ ...s, test: 'fail' }))
        const err = data.error || 'Run failed'
        const errorMap = collectFailureErrorMap(data, err)
        if (Object.keys(errorMap).length > 0) {
          highlightNodeErrors(errorMap)
        }
        setStatusMsg(`Run failed: ${err}`)
        setRunStatus({ phase: 'failed', message: `Run failed: ${err}`, metrics: null })
      }
    } catch {
      setStepStatus((s) => ({ ...s, test: 'fail' }))
      setStatusMsg('Preview failed (API offline)')
      setRunStatus({ phase: 'failed', message: 'Run failed (API offline)', metrics: null })
    }
  }, [nodes, edges, workflowMeta, stepStatus, handleCompile, setNodes, clearNodeHighlights, highlightNodeErrors, collectFailureErrorMap])

  // Helper: set evalStatus on all nodes
  const setAllNodeEvalStatus = useCallback((status, error) => {
    setNodes((nds) =>
      nds.map((n) => ({
        ...n,
        data: { ...n.data, evalStatus: status, evalError: error || null },
      }))
    )
  }, [setNodes])

  const handleDeepRun = useCallback(async () => {
    // Auto-validate + compile first
    if (stepStatus.compile !== 'pass') {
      const compiled = await handleCompile()
      if (!compiled) {
        setStatusMsg('Run skipped: fix validation/compile errors first')
        return
      }
    }
    // Abort any in-flight deep run
    if (deepRunAbortRef.current) deepRunAbortRef.current.abort()
    const controller = new AbortController()
    deepRunAbortRef.current = controller
    setWorkflowStage('deep-run')
    setStepStatus((s) => ({ ...s, run: 'running' }))

    const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
    setEvalState({ stages: [], status: 'running', totalTimeMs: null, error: null, benchmarking: null })
    setRightPanelTab('results')
    setStatusMsg('Run: starting...')
    setRunStatus({ phase: 'running', message: 'Run: starting...', metrics: null })

    // Mark all nodes as running
    setAllNodeEvalStatus('running', null)

    try {
      const budget = {
        run_fingerprint: true,
        run_novelty: true,
      }
      if (benchmarkObserved && Object.keys(benchmarkObserved).length > 0) {
        budget.benchmark_observed = benchmarkObserved
      }
      const res = await apiCall(`/api/v1/workflows/evaluate/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflow, budget }),
        signal: controller.signal,
      })

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        // Parse SSE lines
        const lines = buffer.split('\n')
        buffer = lines.pop() // keep incomplete line in buffer

        let eventType = null
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim()
          } else if (line.startsWith('data: ') && eventType) {
            try {
              const payload = JSON.parse(line.slice(6))

              if (eventType === 'run_id') {
                // Store run_id for later REST queries
                setEvalState((prev) => ({ ...prev, runId: payload.run_id }))
              } else if (eventType === 'stage') {
                setEvalState((prev) => {
                  const existing = prev.stages.findIndex((s) => s.stage === payload.stage)
                  const stages = [...prev.stages]
                  if (existing >= 0) {
                    stages[existing] = payload
                  } else {
                    stages.push(payload)
                  }
                  return { ...prev, stages }
                })

                // Update statusbar with current stage
                if (payload.status === 'running') {
                  setRunStatus({ phase: 'running', message: `Run: ${payload.stage}...`, metrics: null })
                }

                // Conversion error: try to highlight the failing node
                if (payload.stage === 'conversion' && payload.status === 'error') {
                  const errMsg = payload.error || ''
                  const nodeMatch = errMsg.match(/node\s+'?([a-zA-Z0-9_:-]+)/i)
                  if (nodeMatch) {
                    setNodes((nds) =>
                      nds.map((n) =>
                        n.id === nodeMatch[1]
                          ? { ...n, data: { ...n.data, evalStatus: 'fail', evalError: errMsg } }
                          : { ...n, data: { ...n.data, evalStatus: 'fail' } }
                      )
                    )
                  } else {
                    setAllNodeEvalStatus('fail', errMsg)
                  }
                }

                // After profiling completes, annotate nodes with per-op data
                if (payload.stage === 'profiling' && payload.status === 'done' && payload.metrics?.op_profiles) {
                  const profileMap = {}
                  for (const op of payload.metrics.op_profiles) {
                    if (op.aria_node_id) {
                      profileMap[op.aria_node_id] = op
                    }
                  }
                  setNodes((nds) =>
                    nds.map((n) =>
                      profileMap[n.id]
                        ? { ...n, data: { ...n.data, profile: profileMap[n.id] } }
                        : n
                    )
                  )
                }

                // Phase 5.1: Routing/Compression telemetry live-sync
                if (payload.stage === 'routing' && payload.status === 'done' && payload.metrics) {
                  const routingMap = {}
                  const compressionMap = {}
                  
                  if (payload.metrics.op_routing) {
                    for (const r of payload.metrics.op_routing) {
                      if (r.aria_node_id) routingMap[r.aria_node_id] = r
                    }
                  }
                  if (payload.metrics.op_compression) {
                    for (const c of payload.metrics.op_compression) {
                      if (c.aria_node_id) compressionMap[c.aria_node_id] = c
                    }
                  }

                  setNodes((nds) =>
                    nds.map((n) => {
                      let nextData = { ...n.data }
                      if (routingMap[n.id]) nextData.routing = routingMap[n.id]
                      if (compressionMap[n.id]) nextData.compression = compressionMap[n.id]
                      return { ...n, data: nextData }
                    })
                  )
                }

                // Sandbox failure: mark all nodes as failed with the sandbox error
                if (payload.stage === 'sandbox' && payload.status === 'done' && payload.metrics && !payload.metrics.passed) {
                  setAllNodeEvalStatus('fail', 'Sandbox evaluation failed')
                }
              } else if (eventType === 'done') {
                const succeeded = payload.status === 'success'
                setStepStatus((s) => ({ ...s, run: succeeded ? 'pass' : 'fail' }))
                const benchmarking = payload.benchmarking || payload.result?.benchmarking || null
                const summary = benchmarking?.summary || null
                const scoreText = summary?.score != null ? `, score ${Number(summary.score).toFixed(2)}` : ''
                const baselineLossText = importedBaseline?.lossRatio != null
                  ? `, baseline LR ${Number(importedBaseline.lossRatio).toFixed(4)}`
                  : ''
                const doneMsg = succeeded
                  ? `Run complete (${(payload.total_time_ms / 1000).toFixed(1)}s${scoreText}${baselineLossText})`
                  : `Run failed: ${payload.error || payload.status}`
                setEvalState((prev) => ({
                  ...prev,
                  status: payload.status,
                  totalTimeMs: payload.total_time_ms,
                  error: payload.error || null,
                  benchmarking: benchmarking || prev.benchmarking || null,
                }))
                setStatusMsg(doneMsg)
                setRunStatus({
                  phase: succeeded ? 'success' : 'failed',
                  message: doneMsg,
                  metrics: succeeded && summary ? {
                    score: Number(summary.score || 0).toFixed(2),
                    onTarget: summary.on_target || 0,
                    offTarget: summary.off_target || 0,
                  } : null,
                })

                // Final node status
                if (succeeded) {
                  setAllNodeEvalStatus('pass', null)
                } else if (payload.error_stage) {
                  // Error at a specific stage — only mark fail if not already set
                  setNodes((nds) =>
                    nds.map((n) => ({
                      ...n,
                      data: {
                        ...n.data,
                        evalStatus: n.data.evalStatus === 'fail' ? 'fail' : 'fail',
                        evalError: n.data.evalError || payload.error || null,
                      },
                    }))
                  )
                }
              }
            } catch {
              // ignore malformed JSON
            }
            eventType = null
          }
        }
      }

      // Process any remaining data left in buffer after stream ends
      if (buffer.trim()) {
        const remaining = buffer.split('\n')
        let eventType = null
        for (const line of remaining) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim()
          } else if (line.startsWith('data: ') && eventType) {
            try {
              const payload = JSON.parse(line.slice(6))
              if (eventType === 'done') {
                const succeeded = payload.status === 'success'
                setStepStatus((s) => ({ ...s, run: succeeded ? 'pass' : 'fail' }))
                const benchmarking = payload.benchmarking || payload.result?.benchmarking || null
                const summary = benchmarking?.summary || null
                const scoreText = summary?.score != null ? `, score ${Number(summary.score).toFixed(2)}` : ''
                const baselineLossText = importedBaseline?.lossRatio != null
                  ? `, baseline LR ${Number(importedBaseline.lossRatio).toFixed(4)}`
                  : ''
                const doneMsg = succeeded
                  ? `Run complete (${(payload.total_time_ms / 1000).toFixed(1)}s${scoreText}${baselineLossText})`
                  : `Run failed: ${payload.error || payload.status}`
                setEvalState((prev) => ({
                  ...prev,
                  status: payload.status,
                  totalTimeMs: payload.total_time_ms,
                  error: payload.error || null,
                  benchmarking: benchmarking || prev.benchmarking || null,
                }))
                setStatusMsg(doneMsg)
                setRunStatus({
                  phase: succeeded ? 'success' : 'failed',
                  message: doneMsg,
                  metrics: succeeded && summary ? {
                    score: Number(summary.score || 0).toFixed(2),
                    onTarget: summary.on_target || 0,
                    offTarget: summary.off_target || 0,
                  } : null,
                })
                if (succeeded) {
                  setAllNodeEvalStatus('pass', null)
                }
              }
            } catch {
              // ignore malformed JSON
            }
            eventType = null
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        setStepStatus((s) => ({ ...s, run: 'idle' }))
        setStatusMsg('Run cancelled')
        setRunStatus({ phase: 'idle', message: 'Run cancelled', metrics: null })
        setAllNodeEvalStatus(null, null)
      } else {
        setStepStatus((s) => ({ ...s, run: 'fail' }))
        setStatusMsg(`Run failed: ${err.message}`)
        setEvalState((prev) => ({ ...prev, status: 'error', error: err.message }))
        setRunStatus({ phase: 'failed', message: `Run failed: ${err.message}`, metrics: null })
        setAllNodeEvalStatus('fail', err.message)
      }
    }
  }, [nodes, edges, workflowMeta, stepStatus, handleCompile, setNodes, setAllNodeEvalStatus, importedBaseline, benchmarkObserved])

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
  }, [nodes, edges, workflowMeta])

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
  }, [nodes, edges, workflowMeta])

  const loadWorkflowJson = useCallback((workflow) => {
    if (!workflow || workflow.schema_version !== 'workflow_graph.v1') {
      setStatusMsg('Invalid workflow file')
      return
    }

    console.log('[Designer] loadWorkflowJson:', workflow.name,
      (workflow.nodes || []).length, 'nodes,',
      (workflow.edges || []).length, 'edges,',
      components.length, 'components in registry')
    const compIndex = new Map(components.map((c) => [c.id, c]))

    // Build port maps from edges so we can create handles even when
    // the component registry is unavailable (API offline).
    const targetPorts = {}  // nodeId -> Set of port names used as inputs
    const sourcePorts = {}  // nodeId -> Set of port names used as outputs
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

      // Derive a human-readable label: prefer component name, fall back to compId.
      // Never use the raw node ID as a label.
      let label = comp?.name || compId
      if (label === 'unknown' && n.id) {
        // Last resort: clean up the node ID into something readable
        label = n.id.replace(/^node_/, '').replace(/_/g, ' ')
      }

      // If comp not found, derive ports from edges (ensure at least x/y defaults)
      const inputPortNames = targetPorts[n.id] || new Set()
      if (inputPortNames.size === 0 && compId !== 'input' && compId !== 'graph_input') inputPortNames.add('x')
      const outputPortNames = sourcePorts[n.id] || new Set()
      if (outputPortNames.size === 0 && compId !== 'output_head' && compId !== 'graph_output') outputPortNames.add('y')
      const fallbackInputs = [...inputPortNames].map(
        (name) => ({ name, dtype: 'tensor' })
      )
      const fallbackOutputs = [...outputPortNames].map(
        (name) => ({ name, dtype: 'tensor' })
      )

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

    const maxId = nextNodes.reduce((max, n) => {
      // Match both "n_0" and "node_0" formats
      const m = n.id.match(/(?:^n_|^node_)(\d+)$/)
      if (!m) return max
      return Math.max(max, Number(m[1]))
    }, nodeIdCounter)
    nodeIdCounter = Math.max(nodeIdCounter, maxId)

    // Preserve workflow identity (id, name, metadata) so save works correctly.
    setWorkflowMeta({
      workflow_id: workflow.workflow_id || null,
      name: workflow.name || null,
      metadata: workflow.metadata || {},
    })

    // Set nodes first, then defer edges so React Flow has time to
    // measure node dimensions and register handle positions.
    setNodes(normalizeNodePlacement(nextNodes, { grid: [15, 15] }))
    setEdges([])
    setSelectedNodeId(nextNodes[0]?.id || null)
    // Two frames: first for React to commit nodes, second for RF to measure
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        setEdges(nextEdges)
        setStatusMsg(`Loaded workflow: ${workflow.name || workflow.workflow_id}`)
        // Third frame: fit view after edges are rendered (important for embedded)
        requestAnimationFrame(() => {
          fitView({ padding: 0.15, duration: 200 })
        })
      })
    })
  }, [components, setEdges, setNodes, fitView])

  loadWorkflowJsonRef.current = loadWorkflowJson
  handleValidateRef.current = handleValidate
  handlePreviewRef.current = handlePreview
  handleSaveRef.current = handleSave

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
  }, [loadWorkflowJson])

  const handleLoadExample = useCallback(async (path) => {
    if (!path) return
    try {
      // Resolve against current app URL so examples load in both:
      // - standalone dev server (e.g. http://127.0.0.1:5174/)
      // - embedded proxy path (e.g. /designer-proxy/)
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
  }, [loadWorkflowJson])

  useEffect(() => {
    const handler = (e) => {
      if (e.detail) handleLoadExample(e.detail)
    }
    window.addEventListener('load-example', handler)
    return () => window.removeEventListener('load-example', handler)
  }, [handleLoadExample])

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
  }, [])

  const handleClearCanvas = useCallback(() => {
    if (window.confirm('Are you sure you want to clear the entire canvas? This cannot be undone.')) {
      setNodes([])
      setEdges([])
      setSelectedNodeId(null)
      setWorkflowStage('idle')
      setStatusMsg('Canvas cleared')
      setRunStatus({ phase: 'idle', message: 'Idle', metrics: null })
    }
  }, [setNodes, setEdges])

  const getAlignmentTargets = useCallback(() => {
    const selected = nodes.filter((n) => n.selected).map((n) => n.id)
    if (selected.length >= 2) return selected
    return nodes.map((n) => n.id)
  }, [nodes])

  const handleAlignHorizontal = useCallback(() => {
    const targets = getAlignmentTargets()
    if (targets.length < 2) {
      setStatusMsg('Align Horizontal needs at least 2 nodes (select nodes or use full canvas)')
      return
    }
    setNodes((nds) => alignNodesHorizontally(nds, targets, { grid: [15, 15] }))
    setStatusMsg(`Aligned ${targets.length} node(s) horizontally`)
  }, [getAlignmentTargets, setNodes])

  const handleAlignVertical = useCallback(() => {
    const targets = getAlignmentTargets()
    if (targets.length < 2) {
      setStatusMsg('Align Vertical needs at least 2 nodes (select nodes or use full canvas)')
      return
    }
    setNodes((nds) => alignNodesVertically(nds, targets, { grid: [15, 15] }))
    setStatusMsg(`Aligned ${targets.length} node(s) vertically`)
  }, [getAlignmentTargets, setNodes])

  const handleDistributeHorizontal = useCallback(() => {
    const targets = getAlignmentTargets()
    if (targets.length < 2) {
      setStatusMsg('Distribute Horizontal needs at least 2 nodes')
      return
    }
    setNodes((nds) => distributeNodesHorizontally(nds, targets, { grid: [15, 15] }))
    setStatusMsg(`Distributed ${targets.length} node(s) horizontally`)
  }, [getAlignmentTargets, setNodes])

  const handleDistributeVertical = useCallback(() => {
    const targets = getAlignmentTargets()
    if (targets.length < 2) {
      setStatusMsg('Distribute Vertical needs at least 2 nodes')
      return
    }
    setNodes((nds) => distributeNodesVertically(nds, targets, { grid: [15, 15] }))
    setStatusMsg(`Distributed ${targets.length} node(s) vertically`)
  }, [getAlignmentTargets, setNodes])

  const handleTidySelection = useCallback(() => {
    const selected = nodes.filter((n) => n.selected).map((n) => n.id)
    if (selected.length === 0) {
      setStatusMsg('Select one or more nodes, then click Tidy Sel')
      pushToast('Select one or more nodes for Tidy Sel', 'warn')
      return
    }
    setNodes((nds) => tidySelectedNodes(nds, selected, { grid: [15, 15] }))
    setStatusMsg(`Tidied ${selected.length} selected node(s)`)
    pushToast(`Tidied ${selected.length} selected node(s)`, 'success')
  }, [nodes, setNodes, pushToast])

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
  }, [nodes, edges, workflowMeta])

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

    // Step 1: Ask the backend to generate a patch proposal with proper wiring.
    // The generate-patch endpoint analyses the graph, builds ops with edges,
    // saves the workflow to DB (if needed), and creates a proposal.
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

      // Step 2: Apply the proposal. The backend loads the workflow from DB,
      // applies the ops (which include edges), runs _auto_connect fallback,
      // validates, and returns the fully patched workflow.
      setStatusMsg('Applying patch...')
      const applyRes = await apiCall(`/api/v1/aria/apply-patch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ proposal_id: genData.proposal_id, approved_by: 'user' }),
      })
      const applyData = await applyRes.json().catch(() => ({}))
      console.log('[Ask Aria] apply-patch response:', {
        ok: applyRes.ok,
        applied: applyData.applied,
        hasWorkflow: !!applyData.patched_workflow,
        nodeCount: applyData.patched_workflow?.nodes?.length,
        edgeCount: applyData.patched_workflow?.edges?.length,
        edges: applyData.patched_workflow?.edges,
        schema: applyData.patched_workflow?.schema_version,
      })
      if (applyRes.ok && applyData.applied && applyData.patched_workflow) {
        setShowAskAriaModal(false)
        loadWorkflowJsonRef.current?.(applyData.patched_workflow)
        const fp = applyData.new_fingerprint
          ? ` (fingerprint: ${applyData.new_fingerprint.slice(0, 8)}…)`
          : ''
        setStatusMsg(`Patch applied: ${applyData.ops_applied} operations${fp}`)
        setAriaLoading(false)
        return
      }

      // apply-patch returned but without patched_workflow — fall through
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

      // Apply failed but we have a proposal — show error with detail
      const errDetail = applyData.detail || applyData.error || 'Apply returned no patched workflow'
      throw new Error(errDetail)
    } catch (err) {
      console.error('[Ask Aria]', err)
      setStatusMsg(`Aria patch failed: ${err.message || err}`)
      setAriaLoading(false)
    }
  }, [nodes, edges, workflowMeta, evalState, proposalQuery])

  // Patch Application
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
        ? ` (fingerprint: ${data.new_fingerprint.slice(0, 8)}…)`
        : ''
      setStatusMsg(`Patch applied: ${data.ops_applied} operations, saved as v${data.new_version}${fp}`)
      setProposals((prev) => prev.filter(p => p.id !== proposalId))

      // Reload the patched workflow onto the canvas
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
  }, [])

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
  }, [])

  const handlePreviewPatch = useCallback((patch) => {
    setPreviewPatch(patch)
    let ops = []
    try {
      ops = (JSON.parse(patch.patch_json).ops || [])
    } catch {
      setStatusMsg('Proposal preview unavailable: invalid patch payload')
    }
    const affectedNodeIds = ops.filter(op => op.node_id).map(op => op.node_id)
    
    // Highlight nodes in the UI
    setNodes(nds => nds.map(n => ({
      ...n,
      className: affectedNodeIds.includes(n.id) ? 'patch-preview-highlight' : ''
    })))
  }, [setNodes])

  const handleDeleteSelection = useCallback(() => {
    const selectedNodes = nodes.filter((n) => n.selected)
    const selectedEdges = edges.filter((e) => e.selected)
    if (selectedNodes.length === 0 && selectedEdges.length === 0) return false
    deleteElements({ nodes: selectedNodes, edges: selectedEdges })
    if (selectedNodes.some((n) => n.id === selectedNodeId)) {
      setSelectedNodeId(null)
    }
    setStatusMsg(`Deleted ${selectedNodes.length} node(s), ${selectedEdges.length} edge(s)`)
    pushToast(`Deleted selection (${selectedNodes.length} node(s), ${selectedEdges.length} edge(s))`, 'info')
    return true
  }, [nodes, edges, deleteElements, selectedNodeId, pushToast])

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e) => {
      // Skip if typing in an input
      const inInput = e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT'

      // Delete / Backspace — remove selected nodes and edges
      if ((e.key === 'Delete' || e.key === 'Backspace') && !inInput) {
        if (handleDeleteSelection()) {
          e.preventDefault()
        }
        return
      }

      // ? key toggles shortcuts overlay
      if (e.key === '?' && !e.ctrlKey && !e.metaKey) {
        if (inInput) return
        e.preventDefault()
        setShowShortcuts((v) => !v)
        return
      }
      if (e.key === 'Escape') {
        setShowShortcuts(false)
        setSelectedNodeId(null)
        return
      }
      // Ctrl+Z / Cmd+Z — undo
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        undoGraph()
        return
      }
      // Ctrl+Shift+Z or Ctrl+Y — redo
      if (((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'z') || ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y')) {
        e.preventDefault()
        redoGraph()
        return
      }
      // Arrow keys — nudge selected nodes
      if (!inInput && ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) {
        const selectedCount = nodes.filter((n) => n.selected).length
        if (selectedCount > 0) {
          e.preventDefault()
          const step = snapToGridEnabled ? (e.shiftKey ? 45 : 15) : (e.shiftKey ? 20 : 5)
          const dx = e.key === 'ArrowLeft' ? -step : e.key === 'ArrowRight' ? step : 0
          const dy = e.key === 'ArrowUp' ? -step : e.key === 'ArrowDown' ? step : 0
          setNodes((nds) =>
            nds.map((n) => (
              n.selected
                ? { ...n, position: { x: n.position.x + dx, y: n.position.y + dy } }
                : n
            ))
          )
        }
        return
      }
      // Ctrl+S — save
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault()
        handleSave()
        return
      }
      // Ctrl+Enter — compile + run
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault()
        handleCompile().then(() => handlePreview())
        return
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [handleSave, handleCompile, handlePreview, handleDeleteSelection, nodes, setNodes, snapToGridEnabled, undoGraph, redoGraph])

  const isCanvasEmpty = nodes.length === 0
  const selectedNodesCount = useMemo(() => nodes.filter((n) => n.selected).length, [nodes])
  const selectedEdgesCount = useMemo(() => edges.filter((e) => e.selected).length, [edges])
  const hasSelection = selectedNodesCount > 0 || selectedEdgesCount > 0
  const canvasIssue = useMemo(() => {
    if (runStatus.phase === 'failed') {
      return {
        tone: 'fail',
        message: runStatus.message || 'Run failed. Review errors before continuing.',
      }
    }
    if (validateUi.last === 'fail' && validateUi.issues > 0) {
      return {
        tone: 'warn',
        message: `${validateUi.issues} validation issue(s) detected. Fix highlighted nodes and re-run Validate.`,
      }
    }
    return null
  }, [runStatus.phase, runStatus.message, validateUi.last, validateUi.issues])
  const stepGlyph = (status, busy = false) => {
    if (busy || status === 'running') return '↻'
    if (status === 'pass') return '✓'
    if (status === 'fail') return '✕'
    return '•'
  }
  const lastSavedAtText = saveState?.at ? new Date(saveState.at).toLocaleTimeString() : null

  return (
    <div className={`page ${embeddedMode ? 'embedded-mode' : ''} ${isResizing ? 'is-resizing' : ''}`} style={{ '--right-panel-width': `${rightPanelWidth}px` }}>
      {!embeddedMode && (
        <Palette
          components={components}
          onDragStart={() => {}}
          onBlockedDrag={(comp, reason) => {
            setStatusMsg(`Cannot place ${comp?.name || comp?.id}: ${reason || 'incompatible in current context'}`)
            pushToast(`Cannot place ${comp?.name || comp?.id}: ${reason || 'incompatible in current context'}`, 'warn', 3200)
          }}
          constraints={paletteConstraints}
        />
      )}

      <main className="canvas-wrap">
        {!embeddedMode && (
          <header className="topbar">
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              <AriaAvatar 
                size={50} 
                mood={
                  runStatus.phase === 'success' ? 'triumphant' :
                  runStatus.phase === 'failed' ? 'frustrated' :
                  runStatus.phase === 'running' || runStatus.phase === 'compiling' ? 'excited' :
                  'curious'
                } 
              />
              <div>
                <h1>{embeddedMode ? 'Architecture Viewer' : 'Aria Designer'}</h1>
                {!embeddedMode && <p>Graph authoring workspace for user + Aria co-design</p>}
              </div>
            </div>
            <div className="actions">
              <input
                ref={importInputRef}
                type="file"
                accept="application/json"
                onChange={handleImportFile}
                style={{ display: 'none' }}
              />
            <div className="toolbar-group workflow">
              <button 
                type="button"
                className={`step-btn step-state-${stepStatus.validate} ${workflowStage === 'validate' ? 'active' : ''} ${validateUi.inProgress ? 'busy' : ''}`} 
                onClick={handleValidate}
                disabled={validateUi.inProgress}
                title="Step 1: Verify graph structure, ports, and parameters without execution."
              >
                <span className="step-label-row">
                  <span className={`step-glyph ${(validateUi.inProgress || stepStatus.validate === 'running') ? 'step-glyph-spin' : ''}`}>
                    {stepGlyph(stepStatus.validate, validateUi.inProgress)}
                  </span>
                  <span>{validateUi.inProgress ? 'Step 1: Validating...' : 'Step 1: Validate'}</span>
                </span>
              </button>
              <span className={`step-sep step-sep-${stepStatus.validate}`} aria-hidden="true">▶</span>
              <button 
                type="button"
                className={`step-btn step-state-${stepStatus.compile} ${workflowStage === 'compile' ? 'active' : ''}`} 
                onClick={handleCompile}
                title="Step 2: Convert visual graph into a runnable PyTorch module."
              >
                <span className="step-label-row">
                  <span className={`step-glyph ${stepStatus.compile === 'running' ? 'step-glyph-spin' : ''}`}>{stepGlyph(stepStatus.compile)}</span>
                  <span>Step 2: Compile</span>
                </span>
              </button>
              <span className={`step-sep step-sep-${stepStatus.compile}`} aria-hidden="true">▶</span>
              <button 
                type="button"
                className={`step-btn step-state-${stepStatus.test} ${workflowStage === 'run' ? 'active' : ''}`} 
                onClick={handlePreview}
                disabled={isRunDisabled}
                title={isRunDisabled ? "Fix validation errors to run" : "Step 3: Run forward pass with dummy data to verify shapes and latency."}
              >
                <span className="step-label-row">
                  <span className={`step-glyph ${stepStatus.test === 'running' ? 'step-glyph-spin' : ''}`}>{stepGlyph(stepStatus.test)}</span>
                  <span>Step 3: Test</span>
                </span>
              </button>
              <span className={`step-sep step-sep-${stepStatus.test}`} aria-hidden="true">▶</span>
              <button
                type="button"
                className={`step-btn step-state-${stepStatus.run} ${(workflowStage === 'deep-run' || evalState.status === 'running') ? 'active' : ''}`}
                onClick={handleDeepRun}
                disabled={isRunDisabled}
                title="Step 4: Execute full micro-training pipeline (Stage 1) to evaluate loss ratio and novelty."
              >
                <span className="step-label-row">
                  <span className={`step-glyph ${stepStatus.run === 'running' ? 'step-glyph-spin' : ''}`}>{stepGlyph(stepStatus.run)}</span>
                  <span>Step 4: Run</span>
                </span>
              </button>
            </div>
            <div className="toolbar-group files">
              <div className="arrange-dropdown-wrap">
                <button type="button" onClick={() => setFileMenuOpen((v) => !v)} className={fileMenuOpen ? 'active' : ''} title="File operations">
                  File ▾
                </button>
                {fileMenuOpen && (
                  <div className="arrange-dropdown" onMouseLeave={() => setFileMenuOpen(false)}>
                    <button type="button" onClick={() => { handleSave(); setFileMenuOpen(false) }} disabled={saveState.phase === 'saving'}>
                      {saveState.phase === 'saving' ? 'Saving…' : 'Save'}
                    </button>
                    <hr />
                    <button type="button" onClick={() => { handleExportJson(); setFileMenuOpen(false) }}>Export JSON</button>
                    <button type="button" onClick={() => { handleExportPython(); setFileMenuOpen(false) }}>Export Python</button>
                    <hr />
                    <button type="button" onClick={() => { importInputRef.current?.click(); setFileMenuOpen(false) }}>Import JSON</button>
                    <button type="button" onClick={() => { setShowImportDialog(true); setFileMenuOpen(false) }}>Import Research</button>
                    <hr />
                    {exampleOptions.map((ex) => (
                      <button key={ex.value} type="button" onClick={() => { handleLoadExample(ex.value); setFileMenuOpen(false) }}>
                        Example: {ex.label}
                      </button>
                    ))}
                    <hr />
                    <button type="button" onClick={() => { handleReloadComponents(); setFileMenuOpen(false) }}>Reload Components</button>
                    <button type="button" onClick={() => { handleClearCanvas(); setFileMenuOpen(false) }} style={{ color: '#ff5050' }}>Clear Canvas</button>
                  </div>
                )}
              </div>
              {saveState.phase !== 'idle' && (
                <span
                  className={`save-feedback save-${saveState.phase}`}
                  title={saveState.fingerprint ? `${saveState.message}\n${saveState.fingerprint}` : saveState.message}
                >
                  {saveState.message}
                </span>
              )}
            </div>
            <div className="toolbar-group library">
              <button type="button" onClick={undoGraph} disabled={!historyUi.canUndo} title="Undo (Ctrl+Z)">Undo</button>
              <button type="button" onClick={redoGraph} disabled={!historyUi.canRedo} title="Redo (Ctrl+Shift+Z)">Redo</button>
              <div className="arrange-dropdown-wrap">
                <button type="button" onClick={() => setArrangeOpen((v) => !v)} className={arrangeOpen ? 'active' : ''} title="Arrange and align nodes">
                  Arrange ▾
                </button>
                {arrangeOpen && (
                  <div className="arrange-dropdown" onMouseLeave={() => setArrangeOpen(false)}>
                    <button type="button" onClick={() => { handleAlignHorizontal(); setArrangeOpen(false) }}>Align Horizontal</button>
                    <button type="button" onClick={() => { handleAlignVertical(); setArrangeOpen(false) }}>Align Vertical</button>
                    <button type="button" onClick={() => { handleDistributeHorizontal(); setArrangeOpen(false) }}>Distribute Horizontal</button>
                    <button type="button" onClick={() => { handleDistributeVertical(); setArrangeOpen(false) }}>Distribute Vertical</button>
                    <hr />
                    <button type="button" onClick={() => { handleTidySelection(); setArrangeOpen(false) }}>Tidy Selection</button>
                    <hr />
                    <button type="button" onClick={() => { setSnapToGridEnabled((v) => !v); setArrangeOpen(false) }}>
                      Snap to Grid: {snapToGridEnabled ? 'ON' : 'OFF'}
                    </button>
                  </div>
                )}
              </div>
            </div>
            {workflowStage !== 'idle' && runStatus.phase !== 'idle' && (
              <div className={`validation-banner ${
                ['running', 'compiling'].includes(runStatus.phase) ? 'running'
                : runStatus.phase === 'success' ? 'pass'
                : runStatus.phase === 'failed' ? 'fail'
                : 'running'
              }`}>
                {runStatus.message}
              </div>
            )}
            <div className="toolbar-group ai">
              <div className="arrange-dropdown-wrap">
                <button type="button" onClick={() => setViewMenuOpen((v) => !v)} className={viewMenuOpen ? 'active' : ''} title="View options">
                  View ▾
                </button>
                {viewMenuOpen && (
                  <div className="arrange-dropdown" onMouseLeave={() => setViewMenuOpen(false)}>
                    <button type="button" onClick={() => { setHardwareView((v) => !v); setViewMenuOpen(false) }}>
                      {hardwareView ? '\u2713 ' : '\u2003 '}Hardware View
                    </button>
                    <button type="button" onClick={() => { setHeatmapView((v) => !v); setViewMenuOpen(false) }}>
                      {heatmapView ? '\u2713 ' : '\u2003 '}Heatmap
                    </button>
                    <hr />
                    <button type="button" onClick={() => { setShowShortcuts(true); setViewMenuOpen(false) }}>Keyboard Shortcuts</button>
                  </div>
                )}
              </div>
              <button type="button" className="primary" onClick={() => setShowAskAriaModal(true)}>Ask Aria</button>
            </div>
          </div>
        </header>
        )}

        <div className="canvas" ref={reactFlowWrapper}
          onDragEnter={() => setIsDragging(true)}
          onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget)) setIsDragging(false) }}
        >
          <ReactFlow
            nodes={displayNodes}
            edges={edges}
            nodeTypes={nodeTypes}
            defaultEdgeOptions={defaultEdgeOptions}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeDrag={onNodeDrag}
            onNodeDragStop={onNodeDragStop}
            isValidConnection={isValidConnection}
            onNodeClick={(_, node) => {
              setSelectedNodeId(node.id)
            }}
            onPaneClick={() => setSelectedNodeId(null)}
            onDragOver={onDragOver}
            onDrop={onDrop}
            fitView
            snapToGrid={snapToGridEnabled}
            snapGrid={[15, 15]}
          >
            <MiniMap
              pannable
              zoomable
              style={{
                background: 'rgba(10, 22, 40, 0.92)',
                border: '1px solid rgba(90, 138, 181, 0.45)',
                borderRadius: '8px',
              }}
              maskColor="rgba(7, 16, 28, 0.55)"
              nodeColor="#5a8ab5"
            />
            <Background gap={15} color="rgba(255,255,255,0.12)" />
          </ReactFlow>
          {(() => {
            const viewport = getViewport()
            const zoom = Number(viewport?.zoom) || 1
            const tx = Number(viewport?.x) || 0
            const ty = Number(viewport?.y) || 0
            return (
              <>
                {dragGuides.x != null && (
                  <div
                    className="alignment-guide alignment-guide-vertical"
                    style={{ left: `${dragGuides.x * zoom + tx}px` }}
                  />
                )}
                {dragGuides.y != null && (
                  <div
                    className="alignment-guide alignment-guide-horizontal"
                    style={{ top: `${dragGuides.y * zoom + ty}px` }}
                  />
                )}
              </>
            )
          })()}
          {canvasIssue && (
            <div className={`canvas-issue-banner canvas-issue-${canvasIssue.tone}`}>
              {canvasIssue.message}
            </div>
          )}
          {hasSelection && !isDragging && (
            <div className="canvas-selection-hud" aria-live="polite">
              <span className="canvas-selection-count">
                {selectedNodesCount} node(s), {selectedEdgesCount} edge(s) selected
              </span>
              <div className="canvas-selection-actions">
                <button type="button" onClick={handleAlignHorizontal} disabled={selectedNodesCount < 2}>Align H</button>
                <button type="button" onClick={handleAlignVertical} disabled={selectedNodesCount < 2}>Align V</button>
                <button type="button" onClick={handleTidySelection} disabled={selectedNodesCount < 1}>Tidy</button>
                <button type="button" className="danger" onClick={handleDeleteSelection}>Delete</button>
              </div>
            </div>
          )}
          {isCanvasEmpty && !isDragging && <EmptyState onLoadTemplate={handleLoadExample} />}
          <ZoomControls nodes={nodes} edges={edges} setNodes={setNodes} />
        </div>

        {!embeddedMode && (
          <footer className="statusbar">
            <div className="status-block status-primary">
              <span className={`run-chip run-${runStatus.phase}`}>{runStatus.phase.toUpperCase()}</span>
              <span className="status-msg">{runStatus.message}</span>
            </div>
            <div className="status-block status-metrics">
              {runStatus.metrics && Object.entries(runStatus.metrics).map(([k, v]) => (
                <span key={k} className="status-metric">{k}: {v}</span>
              ))}
            </div>
            <div className="status-block status-secondary">
              {statusMsg && <span className="status-msg secondary-msg">{statusMsg}</span>}
              <span className="status-kbd-hint">Keys: ? help · Ctrl/Cmd+S save · Ctrl/Cmd+Enter run</span>
              {lastSavedAtText && <span className="status-updated-chip">Saved {lastSavedAtText}</span>}
              <span className="status-count">{nodes.length} nodes</span>
              <span className="status-count">{edges.length} edges</span>
            </div>
          </footer>
        )}
        <div className="toast-stack" aria-live="polite" aria-atomic="false">
          {toasts.map((t) => (
            <div key={t.id} className={`toast toast-${t.tone || 'info'}`}>
              {t.message}
            </div>
          ))}
        </div>
      </main>

      {!embeddedMode && (
        <aside className="panel right">
          <div 
            className={`resize-handle-left ${isResizing ? 'resizing' : ''}`} 
            onMouseDown={startResizing}
            onKeyDown={handleResizeKeyDown}
            role="separator"
            aria-orientation="vertical"
            aria-label="Resize properties panel"
            aria-valuemin={250}
            aria-valuemax={900}
            aria-valuenow={Math.round(rightPanelWidth)}
            tabIndex={0}
            title="Drag to resize properties panel"
          />
          <div className="panel-tabs">
            <button
              type="button"
              className={rightPanelTab === 'inspector' ? 'active' : ''}
              aria-pressed={rightPanelTab === 'inspector'}
              onClick={() => {
                setRightPanelTab('inspector')
                setPreviewPatch(null)
                setNodes(nds => nds.map(n => ({ ...n, className: '' })))
              }}
            >
              Properties
            </button>
            {scopedProposals.length > 0 && (
              <button
                type="button"
                className={rightPanelTab === 'proposals' ? 'active' : ''}
                aria-pressed={rightPanelTab === 'proposals'}
                onClick={() => setRightPanelTab('proposals')}
              >
                Proposals ({scopedProposals.length})
              </button>
            )}
            <button
              type="button"
              className={rightPanelTab === 'results' ? 'active' : ''}
              aria-pressed={rightPanelTab === 'results'}
              onClick={() => setRightPanelTab('results')}
            >
              Results
            </button>
          </div>

          {rightPanelTab === 'results' ? (
            <RunResultsPanel
              evalState={evalState}
              baseline={importedBaseline}
              benchmarkObserved={benchmarkObserved}
              onBenchmarkObservedChange={setBenchmarkObserved}
            />
          ) : rightPanelTab === 'proposals' ? (
            <PatchPanel
              proposals={scopedProposals}
              onApply={handleApplyPatch}
              onReject={handleRejectPatch}
              onPreview={handlePreviewPatch}
              onClose={() => {
                setRightPanelTab('inspector')
                setPreviewPatch(null)
                setNodes(nds => nds.map(n => ({ ...n, className: '' })))
              }}
            />
          ) : (
            <ErrorBoundary name="Inspector">
              <Inspector
                selectedNode={selectedNode}
                allComponents={components}
                nodeCount={nodes.length}
                edgeCount={edges.length}
                onParamChange={onParamChange}
                helpRequest={helpRequest}
              />
            </ErrorBoundary>
          )}
        </aside>
      )}

      <AskAriaModal
       open={showAskAriaModal}
       onClose={() => { setShowAskAriaModal(false); setAriaSuggestions([]) }}
       onSubmitPrompt={handleAskAriaSubmit}
       onSuggest={handleAskAriaSuggest}
       suggestions={ariaSuggestions}
       loading={ariaLoading}
      />

      <NexusCommandPalette
       open={showNexusPalette}
       onClose={() => setShowNexusPalette(false)}
       components={components}
       onAction={handleNexusAction}
      />
      {showShortcuts && <KeyboardShortcuts onClose={() => setShowShortcuts(false)} />}
      
      {showImportDialog && (
        <ImportDialog 
          onImport={(wf) => loadWorkflowJson(wf)} 
          onClose={() => setShowImportDialog(false)} 
        />
      )}
    </div>
  )
}

// Wrap with provider
export default function App() {
  return (
    <ReactFlowProvider>
      <DesignerApp />
    </ReactFlowProvider>
  )
}
