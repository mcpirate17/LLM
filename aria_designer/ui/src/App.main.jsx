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
import AriaChatPanel from './components/AriaChatPanel'
import ZoomControls from './components/ZoomControls'
import EmptyState from './components/EmptyState'
import KeyboardShortcuts from './components/KeyboardShortcuts'
import HelpPanel from './components/HelpPanel'
import NexusCommandPalette from './components/NexusCommandPalette'
import ImportDialog from './components/ImportDialog'
import RunResultsPanel from './components/RunResultsPanel'
import ErrorBoundary from './components/ErrorBoundary'
import TopBar from './components/TopBar'
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

import { useGraphHistory } from './hooks/useGraphHistory'
import { useNodeStatus } from './hooks/useNodeStatus'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'
import { useFileOperations } from './hooks/useFileOperations'
import { useWorkflowPipeline } from './hooks/useWorkflowPipeline'
import { useAriaCoDesign } from './hooks/useAriaCoDesign'
import { useResizablePanel } from './hooks/useResizablePanel'
import { useEmbeddedBridge } from './hooks/useEmbeddedBridge'

const defaultEdgeOptions = {
  type: 'smoothstep',
  animated: false,
  pathOptions: { borderRadius: 20 },
  style: { stroke: '#5a8ab5', strokeWidth: 1.5 },
  markerEnd: {
    type: MarkerType.ArrowClosed,
    color: '#5a8ab5',
    width: 14,
    height: 14,
  },
}

let nodeIdCounter = 100

function DesignerApp() {
  const [nodes, setNodes, onNodesChange] = useNodesState(starterNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(starterEdges)
  const [selectedNodeId, setSelectedNodeId] = useState(null)
  const [components, setComponents] = useState([])
  const [rightPanelTab, setRightPanelTab] = useState('inspector')
  const [statusMsg, setStatusMsg] = useState('')
  const [showNexusPalette, setShowNexusPalette] = useState(false)
  const [workflowMeta, setWorkflowMeta] = useState({ workflow_id: null, name: null, metadata: {} })
  const [saveState, setSaveState] = useState({ phase: 'idle', message: '', version: null, fingerprint: null, at: 0 })

  const [isDragging, setIsDragging] = useState(false)
  const [paletteConstraints, setPaletteConstraints] = useState({})
  const [showShortcuts, setShowShortcuts] = useState(false)
  const [showHelpPanel, setShowHelpPanel] = useState(false)
  const [showImportDialog, setShowImportDialog] = useState(false)
  const [helpRequest, setHelpRequest] = useState(null)
  const [dragGuides, setDragGuides] = useState({ x: null, y: null })
  const [toasts, setToasts] = useState([])

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

  // Resizable right panel
  const { width: rightPanelWidth, isResizing, startResizing, handleResizeKeyDown } = useResizablePanel()

  const initialParams = useMemo(() => new URLSearchParams(window.location.search), [])

  const [benchmarkObserved, setBenchmarkObserved] = useState(() => {
    try {
      const raw = localStorage.getItem('aria-benchmark-observed')
      const parsed = raw ? JSON.parse(raw) : {}
      return parsed && typeof parsed === 'object' ? parsed : {}
    } catch { return {} }
  })

  const reactFlowWrapper = useRef(null)
  const importInputRef = useRef(null)
  const { screenToFlowPosition, deleteElements, fitView, getViewport } = useReactFlow()

  const currentWorkflowId = workflowMeta?.workflow_id || null
  const importedBaseline = useMemo(() => {
    const meta = workflowMeta?.metadata || {}
    const resultId = String(meta.result_id || '').trim()
    if (!resultId) return null
    const toNum = (v) => { const n = Number(v); return Number.isFinite(n) ? n : null }
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
    const qs = new URLSearchParams({ status: 'pending', workflow_id: String(currentWorkflowId), fresh_only: '1' })
    return `/api/v1/aria/proposals?${qs.toString()}`
  }, [currentWorkflowId])

  // --- Extracted hooks ---

  const pushToast = useCallback((message, tone = 'info', ttl = 2600) => {
    const id = `t_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`
    setToasts((prev) => [...prev, { id, message, tone }])
    window.setTimeout(() => { setToasts((prev) => prev.filter((t) => t.id !== id)) }, Math.max(1200, ttl))
  }, [])

  const { captureSnapshot, undoGraph, redoGraph, historyUi, skipHistoryRef, historyRef } =
    useGraphHistory(setNodes, setEdges, pushToast)

  const { clearNodeHighlights, highlightNodeErrors, collectFailureErrorMap, setAllNodeEvalStatus } =
    useNodeStatus(nodes, setNodes)

  const {
    workflowStage, setWorkflowStage,
    evalState, setEvalState,
    validateUi, stepStatus,
    runStatus, setRunStatus,
    handleValidate, handleCompile, handlePreview, handleDeepRun,
    onParamChange,
  } = useWorkflowPipeline({
    nodes, edges, setNodes, workflowMeta,
    clearNodeHighlights, highlightNodeErrors, collectFailureErrorMap, setAllNodeEvalStatus,
    setStatusMsg, setRightPanelTab,
    importedBaseline, benchmarkObserved,
  })

  const {
    loadWorkflowJson, loadWorkflowJsonRef,
    handleImportFile, handleLoadExample,
    handleExportJson, handleExportPython,
    handleReloadComponents, handleClearCanvas,
    handleSave,
  } = useFileOperations({
    nodes, edges, setNodes, setEdges,
    components, setComponents,
    workflowMeta, setWorkflowMeta,
    setSelectedNodeId, setStatusMsg, setWorkflowStage, setRunStatus,
    fitView, setSaveState,
    getNodeIdCounter: () => nodeIdCounter,
    setNodeIdCounter: (v) => { nodeIdCounter = v },
  })

  const {
    ariaSuggestions, setAriaSuggestions,
    ariaLoading,
    proposals, setProposals,
    previewPatch, setPreviewPatch,
    showAskAriaModal, setShowAskAriaModal,
    handleApplyPatchRef,
    handleAskAriaSuggest, handleAskAriaSubmit,
    handleApplyPatch, handleRejectPatch, handlePreviewPatch,
    handleGhostClick,
  } = useAriaCoDesign({
    nodes, edges, workflowMeta, evalState,
    setNodes, setStatusMsg, setRightPanelTab,
    loadWorkflowJsonRef, pushToast, proposalQuery,
  })

  const scopedProposals = useMemo(
    () => (Array.isArray(proposals) ? proposals.filter((p) => p?.workflow_id === currentWorkflowId) : []),
    [proposals, currentWorkflowId]
  )

  // Store refs for handlers used by NexusCommandPalette and other callbacks
  const handleValidateRef = useRef(null)
  const handlePreviewRef = useRef(null)
  const handleSaveRef = useRef(null)
  handleValidateRef.current = handleValidate
  handlePreviewRef.current = handlePreview
  handleSaveRef.current = handleSave

  const formatImportError = useCallback((payload, fallback = 'Unknown error') => {
    if (!payload) return fallback
    if (typeof payload === 'string') return payload
    if (typeof payload.error === 'string' && payload.error) return payload.error
    if (typeof payload.detail === 'string' && payload.detail) return payload.detail
    if (payload.detail && typeof payload.detail === 'object') {
      const detail = payload.detail
      const issue = Array.isArray(detail.issues) && detail.issues.length > 0 ? detail.issues[0] : null
      if (issue?.message) return `${detail.message || 'Import failed'} (${issue.message})`
      if (detail.message) return detail.message
    }
    if (payload.message) return String(payload.message)
    return fallback
  }, [])

  const { embeddedMode, readOnly, importResultIntoCanvas, postToParent } = useEmbeddedBridge({
    nodes, edges, workflowMeta,
    setStatusMsg, loadWorkflowJsonRef, formatImportError,
  })

  // --- Effects ---

  // Fetch components
  useEffect(() => {
    const ac = new AbortController()
    apiCall(`/api/v1/components?status=approved`, { signal: ac.signal })
      .then((r) => r.json())
      .then((data) => { setComponents(data); setStatusMsg(`${data.length} components loaded`) })
      .catch(() => {
        if (ac.signal.aborted) return
        setStatusMsg('API offline — using mock palette')
        import('./mockData').then((m) => {
          if (ac.signal.aborted) return
          setComponents(m.palette.map((p) => ({
            id: p.id, name: p.label, category: p.category,
            inputs: [{ name: 'x', dtype: 'tensor' }],
            outputs: [{ name: 'y', dtype: 'tensor' }],
          })))
        })
      })
    return () => ac.abort()
  }, [])

  // Fetch proposals
  useEffect(() => {
    if (!proposalQuery || rightPanelTab !== 'proposals') return undefined
    let cancelled = false
    let timer = null
    const fetchProposals = async () => {
      if (cancelled) return
      try {
        const r = await apiCall(proposalQuery)
        const data = await r.json()
        setProposals(Array.isArray(data) ? data : [])
        if (!cancelled) timer = setTimeout(fetchProposals, 5000)
      } catch {
        setProposals([])
        if (!cancelled) timer = setTimeout(fetchProposals, 30000)
      }
    }
    fetchProposals()
    return () => { cancelled = true; if (timer) clearTimeout(timer) }
  }, [proposalQuery, rightPanelTab, setProposals])

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
        if (res.ok) setPaletteConstraints(await res.json())
      } catch {}
    }
    const timer = setTimeout(fetchConstraints, 300)
    return () => clearTimeout(timer)
  }, [nodes, edges, workflowMeta, selectedNodeId])

  useEffect(() => {
    try { localStorage.setItem('aria-benchmark-observed', JSON.stringify(benchmarkObserved || {})) } catch {}
  }, [benchmarkObserved])

  // History capture
  useEffect(() => {
    if (historyRef.current.length > 0) return
    captureSnapshot(nodes, edges)
  }, [captureSnapshot, nodes, edges, historyRef])

  useEffect(() => {
    if (skipHistoryRef.current) return
    const timer = setTimeout(() => {
      if (skipHistoryRef.current) return
      captureSnapshot(nodes, edges)
    }, 220)
    return () => clearTimeout(timer)
  }, [nodes, edges, captureSnapshot, skipHistoryRef])

  // URL param handling
  const urlParamsHandled = useRef(false)
  useEffect(() => {
    if (urlParamsHandled.current || components.length === 0) return
    const resultId = initialParams.get('import_result_id')
    if (resultId) {
      urlParamsHandled.current = true
      importResultIntoCanvas(resultId, { notifyParent: embeddedMode })
    }
  }, [components, importResultIntoCanvas, embeddedMode, initialParams])

  // Ctrl+K command palette
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') { e.preventDefault(); setShowNexusPalette(prev => !prev) }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  // Save state auto-clear
  useEffect(() => {
    if (!saveState?.phase || saveState.phase === 'idle' || saveState.phase === 'saving') return
    const t = setTimeout(() => {
      setSaveState((prev) => (prev.phase === 'saving' ? prev : {
        phase: 'idle', message: '', version: prev.version || null, fingerprint: prev.fingerprint || null, at: prev.at || 0,
      }))
    }, 5000)
    return () => clearTimeout(t)
  }, [saveState])

  // --- Chat handlers ---

  const getWorkflowJsonForChat = useCallback(() => {
    return buildWorkflowJson(nodes, edges, workflowMeta)
  }, [nodes, edges, workflowMeta])

  // --- Canvas event handlers ---

  const selectedNode = useMemo(() => nodes.find((n) => n.id === selectedNodeId) || null, [nodes, selectedNodeId])

  const openNodeHelp = useCallback((nodeId) => {
    setSelectedNodeId(nodeId)
    setRightPanelTab('inspector')
    setHelpRequest({ nodeId, ts: Date.now() })
  }, [])

  // maxFlops via ref: changes on every node mutation, but we don't want that to
  // rebuild nodeTypes (which causes React Flow to unmount/remount ALL nodes).
  // hardwareView/heatmapView are intentional view toggles that SHOULD trigger re-render.
  const maxFlopsRef = useRef(maxFlops)
  maxFlopsRef.current = maxFlops

  const nodeTypes = useMemo(() => ({
    designer: (nodeProps) => (
      <DesignerNode
        {...nodeProps}
        onHelp={() => openNodeHelp(nodeProps.id)}
        hardwareView={hardwareView}
        heatmapView={heatmapView}
        maxFlops={maxFlopsRef.current}
      />
    ),
    ghost: GhostNode,
  }), [openNodeHelp, hardwareView, heatmapView])

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

  // Shared: build node data from a component manifest
  const buildNodeData = useCallback((comp) => ({
    label: comp.name || comp.id, category: comp.category, componentId: comp.id,
    description: comp.description || '', inputs: comp.inputs || [], outputs: comp.outputs || [],
    params: comp.params || {}, paramValues: {}, paramErrors: {},
    performance: comp.performance || {}, manifest: comp,
  }), [])

  // Shared: try to inject a node into the nearest edge (auto-connect)
  const tryEdgeInjection = useCallback((nodeId, nodeData, position, currentNodes, currentEdges, label) => {
    const { edge, distance } = findClosestEdge(position, currentEdges, currentNodes)
    if (!edge || distance >= 25) return
    if (edge.source === nodeId || edge.target === nodeId) return
    const c1 = { source: edge.source, sourceHandle: edge.sourceHandle, target: nodeId, targetHandle: nodeData.inputs[0]?.name || 'x' }
    const c2 = { source: nodeId, sourceHandle: nodeData.outputs[0]?.name || 'y', target: edge.target, targetHandle: edge.targetHandle }
    const filteredPreview = currentEdges.filter((e) => e.id !== edge.id)
    const canInject = validateConnection(c1, currentNodes, filteredPreview)
      && validateConnection(c2, currentNodes, [...filteredPreview, { id: 'tmp_inject', ...c1 }])
    if (!canInject) {
      setStatusMsg(`${label ? `Added ${label}; ` : ''}Skipped auto-insert: incompatible ports`)
      pushToast(`${label ? `Added ${label}; ` : ''}auto-connect skipped`, 'warn')
      return
    }
    setEdges((eds) => {
      const filtered = eds.filter((e) => e.id !== edge.id)
      return [...filtered, { id: `e_inject_1_${Date.now()}`, ...c1 }, { id: `e_inject_2_${Date.now()}`, ...c2 }]
    })
  }, [setEdges, pushToast, setStatusMsg])

  const handleChatApplyPatch = useCallback((p) => {
    if (p.id) {
      handleApplyPatch(p.id)
      return
    }
    if (!p.ops || !p.ops.length) return

    // Apply ops directly to the canvas
    const compIndex = new Map(components.map((c) => [c.id, c]))
    let addedCount = 0
    let wiredCount = 0

    for (const op of p.ops) {
      if (op.op === 'add_node') {
        const payload = op.payload || {}
        const compType = payload.component_type || ''
        const comp = compIndex.get(compType)
        const uiMeta = payload.ui_meta || {}
        const position = {
          x: uiMeta.x ?? (200 + (nodeIdCounter % 5) * 180),
          y: uiMeta.y ?? (100 + (nodeIdCounter % 7) * 120),
        }
        const nid = payload.id || op.node_id || `n_${++nodeIdCounter}`
        const data = comp
          ? { ...buildNodeData(comp), paramValues: payload.params || {} }
          : {
              label: compType, category: 'unknown', componentId: compType,
              description: '', inputs: [{ name: 'x', type: 'tensor' }],
              outputs: [{ name: 'y', type: 'tensor' }],
              params: {}, paramValues: payload.params || {}, paramErrors: {},
              performance: {}, manifest: null,
            }
        const newNode = { id: nid, type: 'designer', position, data }
        setNodes((prev) => [...prev, newNode])
        addedCount++
      } else if (op.op === 'remove_node') {
        const nid = op.node_id
        setNodes((prev) => prev.filter((n) => n.id !== nid))
        setEdges((prev) => prev.filter((e) => e.source !== nid && e.target !== nid))
      } else if (op.op === 'replace_node') {
        const nid = op.node_id
        const newType = op.payload?.component_type
        if (newType) {
          const comp = compIndex.get(newType)
          setNodes((prev) => prev.map((n) => {
            if (n.id !== nid) return n
            const data = comp
              ? { ...buildNodeData(comp), paramValues: op.payload?.params || {} }
              : { ...n.data, label: newType, componentId: newType }
            return { ...n, data }
          }))
        }
      } else if (op.op === 'rewire') {
        const payload = op.payload || {}
        if (payload.remove_edge_id) {
          setEdges((prev) => prev.filter((e) => e.id !== payload.remove_edge_id))
        }
        if (payload.source && payload.target) {
          const edgeId = `e_chat_${Date.now()}_${wiredCount}`
          setEdges((prev) => [...prev, {
            id: edgeId,
            source: payload.source,
            sourceHandle: payload.source_port || 'y',
            target: payload.target,
            targetHandle: payload.target_port || 'x',
          }])
          wiredCount++
        }
      }
    }

    setStatusMsg(`Applied chat patch: ${addedCount} node(s) added, ${wiredCount} edge(s) wired`)
    pushToast(`Patch applied: ${addedCount} nodes, ${wiredCount} edges`, 'ok')
  }, [handleApplyPatch, components, buildNodeData, setNodes, setEdges, setStatusMsg, pushToast])

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge({ ...params }, eds)),
    [setEdges]
  )

  const isValidConnection = useCallback((connection) => {
    return validateConnection(connection, nodes, edges)
  }, [nodes, edges])

  const onNodeDragStop = useCallback(
    (_, node) => {
      const desiredPosition = snapToGridEnabled ? snapPositionToGrid(node.position, [15, 15]) : node.position
      setNodes((nds) => {
        const resolved = findNearestFreePosition(node.id, desiredPosition, nds, { grid: [15, 15] })
        return nds.map((n) => (n.id === node.id ? { ...n, position: resolved } : n))
      })
      tryEdgeInjection(node.id, node.data, desiredPosition, nodes, edges)
      setDragGuides({ x: null, y: null })
    },
    [edges, nodes, setNodes, snapToGridEnabled, tryEdgeInjection]
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
      const xCandidates = [ox, ox + sz.width / 2, ox + sz.width]
      const yCandidates = [oy, oy + sz.height / 2, oy + sz.height]
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
  }, [nodes, getViewport, setNodes])

  const onDragOver = useCallback((e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; setIsDragging(true) }, [])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    const raw = e.dataTransfer.getData('application/aria-component')
    if (!raw) return
    const comp = JSON.parse(raw)
    const compConstraint = paletteConstraints?.[comp.id]
    if (compConstraint?.compatible === false) {
      const reason = Array.isArray(compConstraint.reasons) && compConstraint.reasons.length > 0 ? compConstraint.reasons[0] : 'Current graph state does not allow this component here'
      setStatusMsg(`Cannot place ${comp.name || comp.id}: ${reason}`)
      pushToast(`Cannot place ${comp.name || comp.id}: ${reason}`, 'warn', 3200)
      return
    }
    const position = screenToFlowPosition({ x: e.clientX, y: e.clientY })
    const desiredPos = snapToGridEnabled ? snapPositionToGrid(position, [15, 15]) : position
    const newId = `n_${++nodeIdCounter}`
    const data = buildNodeData(comp)
    const newNode = { id: newId, type: 'designer', position: desiredPos, data }
    setNodes((prev) => normalizeNodePlacement([...prev, newNode], { grid: [15, 15] }))
    setSelectedNodeId(newId)
    tryEdgeInjection(newId, data, desiredPos, [...nodes, newNode], edges, comp.name || comp.id)
  }, [screenToFlowPosition, setNodes, edges, nodes, snapToGridEnabled, paletteConstraints, pushToast, setStatusMsg, buildNodeData, tryEdgeInjection])

  const onAddFromPalette = useCallback((comp) => {
    const newId = `n_${++nodeIdCounter}`
    const position = snapPositionToGrid({ x: 200 + (nodeIdCounter % 5) * 50, y: 100 + (nodeIdCounter % 7) * 60 }, [15, 15])
    const newNode = { id: newId, type: 'designer', position, data: buildNodeData(comp) }
    setNodes((prev) => normalizeNodePlacement([...prev, newNode], { grid: [15, 15] }))
    setSelectedNodeId(newId)
  }, [setNodes, buildNodeData])

  // --- Alignment handlers ---
  const getAlignmentTargets = useCallback(() => {
    const selected = nodes.filter((n) => n.selected).map((n) => n.id)
    return selected.length >= 2 ? selected : nodes.map((n) => n.id)
  }, [nodes])

  const handleAlignHorizontal = useCallback(() => {
    const targets = getAlignmentTargets()
    if (targets.length < 2) { setStatusMsg('Align Horizontal needs at least 2 nodes'); return }
    setNodes((nds) => alignNodesHorizontally(nds, targets, { grid: [15, 15] }))
    setStatusMsg(`Aligned ${targets.length} node(s) horizontally`)
  }, [getAlignmentTargets, setNodes, setStatusMsg])

  const handleAlignVertical = useCallback(() => {
    const targets = getAlignmentTargets()
    if (targets.length < 2) { setStatusMsg('Align Vertical needs at least 2 nodes'); return }
    setNodes((nds) => alignNodesVertically(nds, targets, { grid: [15, 15] }))
    setStatusMsg(`Aligned ${targets.length} node(s) vertically`)
  }, [getAlignmentTargets, setNodes, setStatusMsg])

  const handleDistributeHorizontal = useCallback(() => {
    const targets = getAlignmentTargets()
    if (targets.length < 2) { setStatusMsg('Distribute Horizontal needs at least 2 nodes'); return }
    setNodes((nds) => distributeNodesHorizontally(nds, targets, { grid: [15, 15] }))
    setStatusMsg(`Distributed ${targets.length} node(s) horizontally`)
  }, [getAlignmentTargets, setNodes, setStatusMsg])

  const handleDistributeVertical = useCallback(() => {
    const targets = getAlignmentTargets()
    if (targets.length < 2) { setStatusMsg('Distribute Vertical needs at least 2 nodes'); return }
    setNodes((nds) => distributeNodesVertically(nds, targets, { grid: [15, 15] }))
    setStatusMsg(`Distributed ${targets.length} node(s) vertically`)
  }, [getAlignmentTargets, setNodes, setStatusMsg])

  const handleTidySelection = useCallback(() => {
    const selected = nodes.filter((n) => n.selected).map((n) => n.id)
    if (selected.length === 0) { setStatusMsg('Select nodes first'); pushToast('Select nodes for Tidy Sel', 'warn'); return }
    setNodes((nds) => tidySelectedNodes(nds, selected, { grid: [15, 15] }))
    setStatusMsg(`Tidied ${selected.length} selected node(s)`)
    pushToast(`Tidied ${selected.length} selected node(s)`, 'success')
  }, [nodes, setNodes, pushToast, setStatusMsg])

  const handleDeleteSelection = useCallback(() => {
    const selectedNodes = nodes.filter((n) => n.selected)
    const selectedEdges = edges.filter((e) => e.selected)
    if (selectedNodes.length === 0 && selectedEdges.length === 0) return false
    deleteElements({ nodes: selectedNodes, edges: selectedEdges })
    if (selectedNodes.some((n) => n.id === selectedNodeId)) setSelectedNodeId(null)
    setStatusMsg(`Deleted ${selectedNodes.length} node(s), ${selectedEdges.length} edge(s)`)
    pushToast(`Deleted selection (${selectedNodes.length} node(s), ${selectedEdges.length} edge(s))`, 'info')
    return true
  }, [nodes, edges, deleteElements, selectedNodeId, pushToast, setStatusMsg])

  useKeyboardShortcuts({
    handleSave, handleCompile, handlePreview, handleDeleteSelection,
    undoGraph, redoGraph, setShowShortcuts, setSelectedNodeId,
    nodes, setNodes, snapToGridEnabled,
  })

  const handleNexusAction = useCallback((action) => {
    if (action.id === 'nav-dashboard') { window.location.href = '/research/dashboard' }
    else if (action.id === 'action-validate') { handleValidateRef.current?.() }
    else if (action.id === 'action-run') { handlePreviewRef.current?.() }
    else if (action.id === 'action-save') { handleSaveRef.current?.() }
    else if (action.id.startsWith('add-node-')) {
      const c = action.payload
      setNodes((nds) => [...nds, {
        id: `node_${Date.now()}`, type: 'designer', position: { x: 400, y: 300 },
        data: { label: c.name, category: c.category, component_type: c.id, inputs: c.inputs || [], outputs: c.outputs || [], params: {} },
      }])
      setStatusMsg(`Added ${c.name}`)
    }
  }, [setNodes, setStatusMsg])

  // --- Derived state ---
  const isCanvasEmpty = nodes.length === 0
  const selectedNodesCount = useMemo(() => nodes.filter((n) => n.selected).length, [nodes])
  const selectedEdgesCount = useMemo(() => edges.filter((e) => e.selected).length, [edges])
  const hasSelection = selectedNodesCount > 0 || selectedEdgesCount > 0
  const canvasIssue = useMemo(() => {
    if (runStatus.phase === 'failed') return { tone: 'fail', message: runStatus.message || 'Run failed. Review errors before continuing.' }
    if (validateUi.last === 'fail' && validateUi.issues > 0) return { tone: 'warn', message: `${validateUi.issues} validation issue(s) detected. Fix highlighted nodes and re-run Validate.` }
    return null
  }, [runStatus.phase, runStatus.message, validateUi.last, validateUi.issues])
  const lastSavedAtText = saveState?.at ? new Date(saveState.at).toLocaleTimeString() : null

  // --- Render ---
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
          <TopBar
            runStatus={runStatus} stepStatus={stepStatus} workflowStage={workflowStage}
            validateUi={validateUi}
            saveState={saveState} historyUi={historyUi} snapToGridEnabled={snapToGridEnabled}
            handleValidate={handleValidate} handleCompile={handleCompile}
            handlePreview={handlePreview} handleDeepRun={handleDeepRun}
            handleSave={handleSave} handleExportJson={handleExportJson}
            handleExportPython={handleExportPython} handleImportFile={handleImportFile}
            handleLoadExample={handleLoadExample} handleReloadComponents={handleReloadComponents}
            handleClearCanvas={handleClearCanvas}
            handleAlignHorizontal={handleAlignHorizontal} handleAlignVertical={handleAlignVertical}
            handleDistributeHorizontal={handleDistributeHorizontal} handleDistributeVertical={handleDistributeVertical}
            handleTidySelection={handleTidySelection}
            undoGraph={undoGraph} redoGraph={redoGraph}
            setSnapToGridEnabled={setSnapToGridEnabled} setShowAskAriaModal={setShowAskAriaModal}
            setShowShortcuts={setShowShortcuts} setShowHelpPanel={setShowHelpPanel}
            setShowImportDialog={setShowImportDialog}
            hardwareView={hardwareView} setHardwareView={setHardwareView}
            heatmapView={heatmapView} setHeatmapView={setHeatmapView}
            importInputRef={importInputRef} exampleOptions={exampleOptions} evalState={evalState}
          />
        )}

        <div className="canvas" ref={reactFlowWrapper}
          onDragEnter={() => setIsDragging(true)}
          onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget)) setIsDragging(false) }}
        >
          <ReactFlow
            nodes={nodes} edges={edges} nodeTypes={nodeTypes}
            defaultEdgeOptions={defaultEdgeOptions}
            onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
            onConnect={onConnect} onNodeDrag={onNodeDrag} onNodeDragStop={onNodeDragStop}
            isValidConnection={isValidConnection}
            onNodeClick={(_, node) => setSelectedNodeId(node.id)}
            onPaneClick={() => setSelectedNodeId(null)}
            onDragOver={onDragOver} onDrop={onDrop}
            fitView snapToGrid={snapToGridEnabled} snapGrid={[15, 15]}
          >
            <MiniMap pannable zoomable
              style={{ background: 'rgba(10, 22, 40, 0.92)', border: '1px solid rgba(90, 138, 181, 0.45)', borderRadius: '8px' }}
              maskColor="rgba(7, 16, 28, 0.55)" nodeColor="#5a8ab5"
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
                {dragGuides.x != null && <div className="alignment-guide alignment-guide-vertical" style={{ left: `${dragGuides.x * zoom + tx}px` }} />}
                {dragGuides.y != null && <div className="alignment-guide alignment-guide-horizontal" style={{ top: `${dragGuides.y * zoom + ty}px` }} />}
              </>
            )
          })()}
          {canvasIssue && <div className={`canvas-issue-banner canvas-issue-${canvasIssue.tone}`}>{canvasIssue.message}</div>}
          {hasSelection && !isDragging && (
            <div className="canvas-selection-hud" aria-live="polite">
              <span className="canvas-selection-count">{selectedNodesCount} node(s), {selectedEdgesCount} edge(s) selected</span>
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
            <div key={t.id} className={`toast toast-${t.tone || 'info'}`}>{t.message}</div>
          ))}
        </div>
      </main>

      {!embeddedMode && (
        <aside className="panel right">
          <div
            className={`resize-handle-left ${isResizing ? 'resizing' : ''}`}
            onMouseDown={startResizing} onKeyDown={handleResizeKeyDown}
            role="separator" aria-orientation="vertical" aria-label="Resize properties panel"
            aria-valuemin={250} aria-valuemax={900} aria-valuenow={Math.round(rightPanelWidth)} tabIndex={0}
            title="Drag to resize properties panel"
          />
          <div className="panel-tabs">
            <button type="button" className={rightPanelTab === 'inspector' ? 'active' : ''} aria-pressed={rightPanelTab === 'inspector'}
              onClick={() => { setRightPanelTab('inspector'); setPreviewPatch(null); setNodes(nds => nds.map(n => ({ ...n, className: '' }))) }}>
              Properties
            </button>
            <button type="button" className={rightPanelTab === 'chat' ? 'active' : ''} aria-pressed={rightPanelTab === 'chat'}
              onClick={() => setRightPanelTab('chat')}>
              Aria Chat
            </button>
            {scopedProposals.length > 0 && (
              <button type="button" className={rightPanelTab === 'proposals' ? 'active' : ''} aria-pressed={rightPanelTab === 'proposals'}
                onClick={() => setRightPanelTab('proposals')}>
                Proposals ({scopedProposals.length})
              </button>
            )}
            <button type="button" className={rightPanelTab === 'results' ? 'active' : ''} aria-pressed={rightPanelTab === 'results'}
              onClick={() => setRightPanelTab('results')}>
              Results
            </button>
          </div>

          {rightPanelTab === 'results' ? (
            <RunResultsPanel evalState={evalState} baseline={importedBaseline}
              benchmarkObserved={benchmarkObserved} onBenchmarkObservedChange={setBenchmarkObserved} />
          ) : rightPanelTab === 'chat' ? (
            <ErrorBoundary name="Chat">
              <AriaChatPanel
                workflowJsonFn={getWorkflowJsonForChat}
                onApplyPatch={handleChatApplyPatch}
              />
            </ErrorBoundary>
          ) : rightPanelTab === 'proposals' ? (
            <PatchPanel proposals={scopedProposals} onApply={handleApplyPatch} onReject={handleRejectPatch}
              onPreview={handlePreviewPatch}
              onClose={() => { setRightPanelTab('inspector'); setPreviewPatch(null); setNodes(nds => nds.map(n => ({ ...n, className: '' }))) }} />
          ) : (
            <ErrorBoundary name="Inspector">
              <Inspector selectedNode={selectedNode} allComponents={components} nodeCount={nodes.length}
                edgeCount={edges.length} onParamChange={onParamChange} helpRequest={helpRequest} />
            </ErrorBoundary>
          )}
        </aside>
      )}

      <AskAriaModal
        open={showAskAriaModal}
        onClose={() => { setShowAskAriaModal(false); setAriaSuggestions([]) }}
        onSubmitPrompt={handleAskAriaSubmit}
        onSuggest={handleAskAriaSuggest}
        onSwitchToChat={() => { setShowAskAriaModal(false); setAriaSuggestions([]); setRightPanelTab('chat') }}
        suggestions={ariaSuggestions}
        loading={ariaLoading}
      />
      <NexusCommandPalette open={showNexusPalette} onClose={() => setShowNexusPalette(false)}
        components={components} onAction={handleNexusAction} />
      {showShortcuts && <KeyboardShortcuts onClose={() => setShowShortcuts(false)} />}
      <HelpPanel isOpen={showHelpPanel} onClose={() => setShowHelpPanel(false)} />
      {showImportDialog && (
        <ImportDialog onImport={(wf) => loadWorkflowJson(wf)} onClose={() => setShowImportDialog(false)} />
      )}
    </div>
  )
}

export default function App() {
  return (
    <ReactFlowProvider>
      <DesignerApp />
    </ReactFlowProvider>
  )
}
