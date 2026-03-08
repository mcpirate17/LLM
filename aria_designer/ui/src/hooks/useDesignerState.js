import { useCallback, useState, useRef, useMemo, useEffect } from 'react'
import {
  useNodesState,
  useEdgesState,
  useReactFlow
} from '@xyflow/react'
import { starterEdges, starterNodes } from '../mockData'
import { apiCall } from '../services/apiService'
import { buildWorkflowJson } from '../utils/workflow'
import { normalizeNodePlacement, alignNodesHorizontally, alignNodesVertically, distributeNodesHorizontally, distributeNodesVertically, tidySelectedNodes } from '../utils/layout'

export function useDesignerState() {
  const [nodes, setNodes, onNodesChange] = useNodesState(starterNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(starterEdges)
  const [selectedNodeId, setSelectedNodeId] = useState(null)
  const [components, setComponents] = useState([])
  
  const { fitView } = useReactFlow()
  
  // UI States
  const [rightPanelTab, setRightPanelTab] = useState('inspector')
  const [rightPanelWidth, setRightPanelWidth] = useState(300)
  const [statusMsg, setStatusMsg] = useState('')
  const [toasts, setToasts] = useState([])
  const [isDragging, setIsDragging] = useState(false)
  const [dragGuides, setDragGuides] = useState({ x: null, y: null })
  
  // View Settings
  const [snapToGridEnabled, setSnapToGridEnabled] = useState(true)
  const [hardwareView, setHardwareView] = useState(false)
  const [heatmapView, setHeatmapView] = useState(false)
  const [embeddedMode, setEmbeddedMode] = useState(false)
  
  // Workflow Stage and Status
  const [workflowStage, setWorkflowStage] = useState('idle')
  const [stepStatus, setStepStatus] = useState({ validate: 'idle', compile: 'idle', test: 'idle', run: 'idle' })
  const [runStatus, setRunStatus] = useState({ phase: 'idle', message: 'Idle', metrics: null })
  const [evalState, setEvalState] = useState({ stages: [], status: 'idle', totalTimeMs: null, error: null, benchmarking: null })
  const [workflowMeta, setWorkflowMeta] = useState({ workflow_id: null, name: null, metadata: {} })
  const [benchmarkObserved, setBenchmarkObserved] = useState({})
  const [paletteConstraints, setPaletteConstraints] = useState({})

  // History management
  const historyRef = useRef([])
  const futureRef = useRef([])
  const skipHistoryRef = useRef(false)
  const lastSnapshotSigRef = useRef('')
  const [historyUi, setHistoryUi] = useState({ canUndo: false, canRedo: false })

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

  // Node highlight helpers
  const clearNodeHighlights = useCallback(() => {
    setNodes((nds) => nds.map((n) => ({ ...n, className: '', data: { ...n.data, evalStatus: null, evalError: null } })))
  }, [setNodes])

  const highlightNodeErrors = useCallback((errorMap) => {
    setNodes((nds) =>
      nds.map((n) =>
        errorMap[n.id]
          ? { ...n, className: 'node-invalid', data: { ...n.data, evalStatus: 'fail', evalError: errorMap[n.id][0] } }
          : n
      )
    )
  }, [setNodes])

  const captureSnapshot = useCallback((nextNodes, nextEdges) => {
    if (skipHistoryRef.current) return
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

  // Loader Logic
  const loadWorkflowJson = useCallback((workflow) => {
    if (!workflow || workflow.schema_version !== 'workflow_graph.v1') {
      setStatusMsg('Invalid workflow file')
      return
    }

    const compIndex = new Map(components.map((c) => [c.id, c]))
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
      const inputPortNames = targetPorts[n.id] || new Set()
      if (inputPortNames.size === 0 && compId !== 'input' && compId !== 'graph_input') inputPortNames.add('x')
      const outputPortNames = sourcePorts[n.id] || new Set()
      if (outputPortNames.size === 0 && compId !== 'output_head' && compId !== 'graph_output') outputPortNames.add('y')
      
      return {
        id: n.id,
        type: 'designer',
        position: n.ui_meta?.position || { x: 200, y: 100 },
        data: {
          label,
          category: comp?.category || category,
          componentId: compType,
          inputs: comp?.inputs || [...inputPortNames].map(name => ({ name, dtype: 'tensor' })),
          outputs: comp?.outputs || [...outputPortNames].map(name => ({ name, dtype: 'tensor' })),
          params: comp?.params || {},
          paramValues: n.params || {},
          manifest: comp || {},
        },
      }
    })

    const nextEdges = (workflow.edges || []).map((e, idx) => ({
      id: e.id || `e_${idx}`,
      source: e.source,
      target: e.target,
      sourceHandle: e.source_port || 'y',
      targetHandle: e.target_port || 'x',
    }))

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
          fitView({ padding: 0.15, duration: 200 })
        })
      })
    })
  }, [components, setEdges, setNodes, fitView])

  // Alignment Logic
  const getAlignmentTargets = useCallback(() => {
    const selected = nodes.filter((n) => n.selected).map((n) => n.id)
    return selected.length >= 2 ? selected : nodes.map((n) => n.id)
  }, [nodes])

  const handleAlignHorizontal = useCallback(() => {
    const targets = getAlignmentTargets()
    setNodes((nds) => alignNodesHorizontally(nds, targets, { grid: [15, 15] }))
    setStatusMsg(`Aligned ${targets.length} node(s) horizontally`)
  }, [getAlignmentTargets, setNodes])

  const handleAlignVertical = useCallback(() => {
    const targets = getAlignmentTargets()
    setNodes((nds) => alignNodesVertically(nds, targets, { grid: [15, 15] }))
    setStatusMsg(`Aligned ${targets.length} node(s) vertically`)
  }, [getAlignmentTargets, setNodes])

  // Effects: Fetch components
  useEffect(() => {
    const fetchComps = async () => {
      try {
        const res = await apiCall('/api/v1/components?status=approved')
        const data = await res.json()
        setComponents(data)
      } catch (err) {
        console.warn('Failed to fetch components', err)
      }
    }
    fetchComps()
  }, [])

  // Effects: Autosave
  useEffect(() => {
    if (embeddedMode) return
    const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
    localStorage.setItem('aria-workflow-autosave', JSON.stringify(workflow))
  }, [nodes, edges, workflowMeta, embeddedMode])

  const maxFlops = useMemo(() => {
    let max = 0
    nodes.forEach(n => {
      const f = n.data?.profile?.flops || n.data?.performance?.flops_forward || 0
      if (f > max) max = f
    })
    return max || 1
  }, [nodes])

  return {
    nodes, setNodes, onNodesChange,
    edges, setEdges, onEdgesChange,
    selectedNodeId, setSelectedNodeId,
    components, setComponents,
    rightPanelTab, setRightPanelTab,
    rightPanelWidth, setRightPanelWidth,
    statusMsg, setStatusMsg,
    toasts, setToasts, pushToast,
    isDragging, setIsDragging,
    dragGuides, setDragGuides,
    snapToGridEnabled, setSnapToGridEnabled,
    hardwareView, setHardwareView,
    heatmapView, setHeatmapView,
    embeddedMode, setEmbeddedMode,
    workflowStage, setWorkflowStage,
    stepStatus, setStepStatus,
    runStatus, setRunStatus,
    evalState, setEvalState,
    workflowMeta, setWorkflowMeta,
    benchmarkObserved, setBenchmarkObserved,
    paletteConstraints, setPaletteConstraints,
    historyUi, captureSnapshot,
    clearNodeHighlights, highlightNodeErrors,
    loadWorkflowJson, handleAlignHorizontal, handleAlignVertical,
    maxFlops, skipHistoryRef
  }
}
