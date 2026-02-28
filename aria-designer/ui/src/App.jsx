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
import Palette from './components/Palette'
import Inspector from './components/Inspector'
import PatchPanel from './components/PatchPanel'
import AskAriaModal from './components/AskAriaModal'
import ZoomControls from './components/ZoomControls'
import EmptyState from './components/EmptyState'
import KeyboardShortcuts from './components/KeyboardShortcuts'
import ImportDialog from './components/ImportDialog'
import RunResultsPanel from './components/RunResultsPanel'
import ErrorBoundary from './components/ErrorBoundary'
import { isValidConnection as validateConnection } from './utils/validation'
import { buildWorkflowJson } from './utils/workflow'
import { findClosestEdge } from './utils/geometry'
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
  const [workflowMeta, setWorkflowMeta] = useState({ workflow_id: null, name: null, metadata: {} })
  const [ariaSuggestions, setAriaSuggestions] = useState([])
  const [ariaLoading, setAriaLoading] = useState(false)

  const [isDragging, setIsDragging] = useState(false)
  const [paletteConstraints, setPaletteConstraints] = useState({})
  const [showShortcuts, setShowShortcuts] = useState(false)
  const [showImportDialog, setShowImportDialog] = useState(false)
  const [helpRequest, setHelpRequest] = useState(null)
  
  const [snapToGridEnabled, setSnapToGridEnabled] = useState(true)
  
  const [rightPanelWidth, setRightPanelWidth] = useState(300)
  const [isResizing, setIsResizing] = useState(false)
  const resizeRef = useRef({ startX: 0, startWidth: 300 })

  const startResizing = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    resizeRef.current = { startX: e.clientX, startWidth: rightPanelWidth }
    setIsResizing(true)
  }, [rightPanelWidth])

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
  const reactFlowWrapper = useRef(null)
  const importInputRef = useRef(null)
  const { screenToFlowPosition, deleteElements, fitView } = useReactFlow()

  const importResultIntoCanvas = useCallback(async (resultId, options = {}) => {
    const shouldNotifyParent = Boolean(options.notifyParent)
    const rid = String(resultId || '').trim()
    if (!rid) return
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
      loadWorkflowJsonRef.current?.(wf)
      setStatusMsg(`Loaded architecture: ${wf.name || rid}`)
      if (shouldNotifyParent && window.parent && window.parent !== window) {
        console.log('[Designer] posting graph-loaded to parent for', rid, '(',
          (wf.nodes || []).length, 'nodes,', (wf.edges || []).length, 'edges)')
        window.parent.postMessage({
          source: 'aria-designer',
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
      const message = err?.name === 'AbortError'
        ? 'Import timed out'
        : (err?.message || String(err))
      setStatusMsg(`Failed to import ${rid}: ${message}`)
      if (shouldNotifyParent && window.parent && window.parent !== window) {
        window.parent.postMessage({
          source: 'aria-designer',
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
          data: { ...n.data, errors: errs },
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

  // Fetch proposals from API
  useEffect(() => {
    const fetchProposals = async () => {
      try {
        const r = await apiCall(`/api/v1/aria/proposals?status=pending`)
        const data = await r.json()
        setProposals(data)
      } catch (err) {
        console.error('Failed to fetch proposals', err)
      }
    }
    fetchProposals()
    const timer = setInterval(fetchProposals, 5000) // Poll every 5s
    return () => clearInterval(timer)
  }, [])

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
    window.parent.postMessage({ source: 'aria-designer', type, ...payload }, '*')
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

  // Keep refs for nodes/edges so the message handler can access current
  // values without causing the effect to re-run on every graph change.
  const nodesRef = useRef(nodes)
  const edgesRef = useRef(edges)
  useEffect(() => { nodesRef.current = nodes }, [nodes])
  useEffect(() => { edgesRef.current = edges }, [edges])

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
  useEffect(() => {
    if (!embeddedMode) return
    const handler = (e) => {
      if (e.data?.target !== 'aria-designer') return
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
  }, [embeddedMode, importResultIntoCanvas, postToParent])

  // URL param handling — load a workflow from research pipeline when embedded.
  // Supports: ?import_result_id=res_xxx&readonly=1&embedded=1
  const urlParamsHandled = useRef(false)
  useEffect(() => {
    if (urlParamsHandled.current || components.length === 0) return
    const resultId = initialParams.get('import_result_id')

    if (resultId) {
      urlParamsHandled.current = true
      importResultIntoCanvas(resultId, { notifyParent: embeddedMode })
    }
  }, [components, importResultIntoCanvas, embeddedMode])

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
        />
      ),
    }),
    [openNodeHelp]
  )

  const exampleOptions = useMemo(() => ([
    { label: 'Simple Linear', value: '/examples/simple_linear.json' },
    { label: 'Tropical Attention', value: '/examples/tropical_attention.json' },
    { label: 'Tropical Block', value: '/examples/tropical_block.json' },
    { label: 'Transformer Mini', value: '/examples/transformer_mini.json' },
    { label: 'SSM Stack', value: '/examples/ssm_stack.json' },
    { label: 'Hybrid Attn+SSM+MoE', value: '/examples/hybrid_attn_ssm_moe.json' },
  ]), [])

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge({ ...params }, eds)),
    [setEdges]
  )

  const onNodeDragStop = useCallback(
    (_, node) => {
      // Auto-injection for existing nodes
      const position = node.position
      const { edge, distance } = findClosestEdge(position, edges, nodes)
      
      // Ensure we don't try to inject into an edge that is already connected to this node
      if (edge && distance < 25 && edge.source !== node.id && edge.target !== node.id) {
        setEdges((eds) => {
          const filtered = eds.filter((e) => e.id !== edge.id)
          return [
            ...filtered,
            {
              id: `e_inject_1_${Date.now()}`,
              source: edge.source,
              sourceHandle: edge.sourceHandle,
              target: node.id,
              targetHandle: node.data.inputs[0]?.name || 'x',
            },
            {
              id: `e_inject_2_${Date.now()}`,
              source: node.id,
              sourceHandle: node.data.outputs[0]?.name || 'y',
              target: edge.target,
              targetHandle: edge.targetHandle,
            },
          ]
        })
      }
    },
    [edges, nodes, setEdges]
  )

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
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY })
      const newId = `n_${++nodeIdCounter}`

      const newNode = {
        id: newId,
        type: 'designer',
        position,
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
      setNodes((prev) => [...prev, newNode])
      setSelectedNodeId(newId)

      // Auto-injection logic: check if dropped on/near an edge
      const { edge, distance } = findClosestEdge(position, edges, nodes)
      if (edge && distance < 25) {
        setEdges((eds) => {
          const filtered = eds.filter((e) => e.id !== edge.id)
          return [
            ...filtered,
            {
              id: `e_inject_1_${Date.now()}`,
              source: edge.source,
              sourceHandle: edge.sourceHandle,
              target: newId,
              targetHandle: newNode.data.inputs[0]?.name || 'x',
            },
            {
              id: `e_inject_2_${Date.now()}`,
              source: newId,
              sourceHandle: newNode.data.outputs[0]?.name || 'y',
              target: edge.target,
              targetHandle: edge.targetHandle,
            },
          ]
        })
      }
    },
    [screenToFlowPosition, setNodes, edges, nodes, setEdges]
  )

  const onAddFromPalette = useCallback(
    (comp) => {
      const newId = `n_${++nodeIdCounter}`
      const newNode = {
        id: newId,
        type: 'designer',
        position: { x: 200 + (nodeIdCounter % 5) * 50, y: 100 + (nodeIdCounter % 7) * 60 },
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
      setNodes((prev) => [...prev, newNode])
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
      try {
        const res = await apiCall(`/api/v1/workflows/validate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ workflow }),
        })
        if (!res.ok) throw new Error(`validate ${res.status}`)
        data = await res.json()
      } catch {
        const res = await apiCall(`/api/v1/workflows/validate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ workflow }),
        })
        data = await res.json()
      }

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
      try {
        const res = await apiCall(`/api/v1/workflows/compile`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ workflow, target: 'cpu' }),
        })
        if (!res.ok) throw new Error(`compile ${res.status}`)
        data = await res.json()
      } catch {
        usedDesignerApi = false
        const res = await apiCall(`/api/v1/workflows/compile`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ workflow, target: 'cpu' }),
        })
        data = await res.json()
      }

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
        const nodeMatch = String(err).match(/node\\s+([a-zA-Z0-9_:-]+)/i)
        if (nodeMatch && nodeMatch[1]) {
          highlightNodeErrors({ [nodeMatch[1]]: [err] })
        }
        setStatusMsg(`Compilation failed: ${err}`)
        setRunStatus({ phase: 'failed', message: `Compile failed: ${err}`, metrics: null })
        return false
      }
    } catch {
      setStepStatus((s) => ({ ...s, compile: 'fail' }))
      setStatusMsg('Compilation failed (API offline)')
      setRunStatus({ phase: 'failed', message: 'Compile failed (API offline)', metrics: null })
      return false
    }
  }, [nodes, edges, workflowMeta, stepStatus, handleValidate, clearNodeHighlights, highlightNodeErrors])

  const handleSave = useCallback(async () => {
    const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
    setStatusMsg('Saving workflow...')
    try {
      const res = await apiCall(`/api/v1/workflows/${workflow.workflow_id || 'default'}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(workflow),
      })
      if (!res.ok) throw new Error(`save ${res.status}`)
      const data = await res.json()
      const fp = data.fingerprint ? ` (fingerprint: ${data.fingerprint.slice(0, 8)}...)` : ''
      // Update metadata with fingerprint from server
      if (data.fingerprint) {
        setWorkflowMeta((prev) => ({
          ...prev,
          workflow_id: data.workflow_id || prev.workflow_id,
          metadata: { ...prev.metadata, graph_fingerprint: data.fingerprint },
        }))
      }
      setStatusMsg(`Workflow saved v${data.version}${fp}`)
    } catch {
      // Save to localStorage as fallback
      localStorage.setItem('aria-workflow', JSON.stringify(workflow))
      setStatusMsg('Saved to browser (API offline)')
    }
  }, [nodes, edges, workflowMeta])

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
        const nodeMatch = String(err).match(/node\\s+([a-zA-Z0-9_:-]+)/i)
        if (nodeMatch && nodeMatch[1]) {
          highlightNodeErrors({ [nodeMatch[1]]: [err] })
        }
        setStatusMsg(`Run failed: ${err}`)
        setRunStatus({ phase: 'failed', message: `Run failed: ${err}`, metrics: null })
      }
    } catch {
      setStepStatus((s) => ({ ...s, test: 'fail' }))
      setStatusMsg('Preview failed (API offline)')
      setRunStatus({ phase: 'failed', message: 'Run failed (API offline)', metrics: null })
    }
  }, [nodes, edges, workflowMeta, stepStatus, handleCompile, setNodes, clearNodeHighlights, highlightNodeErrors])

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
        setStatusMsg('Deep Run skipped: fix validation/compile errors first')
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
    setStatusMsg('Deep Run: starting...')
    setRunStatus({ phase: 'running', message: 'Deep Run: starting...', metrics: null })

    // Mark all nodes as running
    setAllNodeEvalStatus('running', null)

    try {
      const res = await apiCall(`/api/v1/workflows/evaluate/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflow, budget: { run_fingerprint: true, run_novelty: true } }),
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
                  setRunStatus({ phase: 'running', message: `Deep Run: ${payload.stage}...`, metrics: null })
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
                const doneMsg = succeeded
                  ? `Deep Run complete (${(payload.total_time_ms / 1000).toFixed(1)}s)`
                  : `Deep Run failed: ${payload.error || payload.status}`
                setEvalState((prev) => ({
                  ...prev,
                  status: payload.status,
                  totalTimeMs: payload.total_time_ms,
                  error: payload.error || null,
                  benchmarking: payload.benchmarking || payload.result?.benchmarking || prev.benchmarking || null,
                }))
                setStatusMsg(doneMsg)
                setRunStatus({
                  phase: succeeded ? 'success' : 'failed',
                  message: doneMsg,
                  metrics: null,
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
    } catch (err) {
      if (err.name === 'AbortError') {
        setStepStatus((s) => ({ ...s, run: 'idle' }))
        setStatusMsg('Deep Run cancelled')
        setRunStatus({ phase: 'idle', message: 'Deep Run cancelled', metrics: null })
        setAllNodeEvalStatus(null, null)
      } else {
        setStepStatus((s) => ({ ...s, run: 'fail' }))
        setStatusMsg(`Deep Run failed: ${err.message}`)
        setEvalState((prev) => ({ ...prev, status: 'error', error: err.message }))
        setRunStatus({ phase: 'failed', message: `Deep Run failed: ${err.message}`, metrics: null })
        setAllNodeEvalStatus('fail', err.message)
      }
    }
  }, [nodes, edges, workflowMeta, stepStatus, handleCompile, setNodes, setAllNodeEvalStatus])

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

      // If comp not found, derive ports from edges
      const fallbackInputs = [...(targetPorts[n.id] || [])].map(
        (name) => ({ name, dtype: 'tensor' })
      )
      const fallbackOutputs = [...(sourcePorts[n.id] || [])].map(
        (name) => ({ name, dtype: 'tensor' })
      )

      return {
        id: n.id,
        type: 'designer',
        position: n.ui_meta?.position || { x: 200, y: 100 },
        data: {
          label: comp?.name || compId,
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

    const nextEdges = (workflow.edges || []).map((e, idx) => ({
      id: e.id || `e_${idx}`,
      source: e.source,
      target: e.target,
      sourceHandle: e.source_port || 'y',
      targetHandle: e.target_port || 'x',
    }))

    const maxId = nextNodes.reduce((max, n) => {
      const m = n.id.match(/n_(\\d+)/)
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
    setNodes(nextNodes)
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
      const res = await apiCall(path)
      const data = await res.json()
      loadWorkflowJson(data)
    } catch {
      setStatusMsg('Failed to load example')
    }
  }, [loadWorkflowJson])

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

  const handleAskAriaSubmit = useCallback(async (promptText) => {
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

    // Preferred path: backend-generated deterministic patch proposal.
    try {
      const res = await apiCall(`/api/v1/aria/generate-patch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflow, prompt: effectivePrompt, base_version: 1 }),
      })
      const data = await res.json()
      if (res.ok && data.proposal_id) {
        setStatusMsg(`Aria proposal created: ${data.proposal_id}`)
        setRightPanelTab('proposals')
        setShowAskAriaModal(false)
        const pRes = await apiCall(`/api/v1/aria/proposals?status=pending`)
        const pData = await pRes.json()
        setProposals(Array.isArray(pData) ? pData : [])
        setAriaLoading(false)
        return
      }
    } catch {
      // Fallback to local heuristic proposal creation below.
    }

    const sourceIds = new Set((workflow.edges || []).map((e) => e.source))
    const sinkNodes = (workflow.nodes || []).filter((n) => !sourceIds.has(n.id))
    const lastNode = sinkNodes[sinkNodes.length - 1] || workflow.nodes[workflow.nodes.length - 1]
    const hasOutput = (workflow.nodes || []).some((n) => String(n.component_type || '').includes('output'))

    const lower = prompt.toLowerCase()
    let componentType = null
    if (lower.includes('output')) componentType = 'io/output_head'
    if (lower.includes('relu')) componentType = 'math/relu'
    if (lower.includes('rmsnorm') || lower.includes('norm')) componentType = 'normalization/rmsnorm'
    if (lower.includes('tropical attention')) componentType = 'math_space/tropical_attention'
    if (lower.includes('tropical gate') || lower.includes('gate')) componentType = 'math_space/tropical_gate'
    if (!componentType && !hasOutput) componentType = 'io/output_head'
    if (!componentType && ariaSuggestions[0]?.component) {
      const c = ariaSuggestions[0].component
      componentType = c.id.includes('/') ? c.id : `${c.category || 'math'}/${c.id}`
    }
    if (!componentType) {
      setStatusMsg('Ask Aria could not infer a concrete component from prompt')
      return
    }

    const newNodeId = `aria_${Date.now().toString(36)}`
    const ops = [{
      op: 'add_node',
      payload: {
        id: newNodeId,
        component_type: componentType,
        params: {},
        ui_meta: { position: { x: 520, y: 220 } },
        edges: lastNode ? [{
          source: lastNode.id,
          source_port: 'y',
          target: newNodeId,
          target_port: 'x',
        }] : [],
      },
    }]

    try {
      const patch = {
        workflow_id: workflow.workflow_id,
        base_version: 1,
        author: 'aria',
        rationale: `Prompt: ${effectivePrompt}`,
        expected_impact: { summary: 'User-directed change proposal' },
        ops,
      }
      const res = await apiCall(`/api/v1/aria/propose-patch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch),
      })
      const data = await res.json()
      if (!res.ok || !data.proposal_id) {
        throw new Error(data.detail || data.error || 'Proposal creation failed')
      }
      setStatusMsg(`Aria proposal created: ${data.proposal_id}`)
      setRightPanelTab('proposals')
      setShowAskAriaModal(false)
      const pRes = await apiCall(`/api/v1/aria/proposals?status=pending`)
      const pData = await pRes.json()
      setProposals(Array.isArray(pData) ? pData : [])
    } catch (err) {
      setStatusMsg(`Failed to create proposal: ${err.message || err}`)
    } finally {
      setAriaLoading(false)
    }
  }, [nodes, edges, workflowMeta, ariaSuggestions, evalState])

  // Patch Application
  const handleApplyPatch = useCallback(async (proposalId) => {
    setStatusMsg('Applying patch...')
    try {
      const res = await apiCall(`/api/v1/aria/apply-patch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ proposal_id: proposalId, approved_by: 'user' }),
      })
      const data = await res.json()
      if (data.applied) {
        const fp = data.new_fingerprint
          ? ` (fingerprint: ${data.new_fingerprint.slice(0, 8)}…)`
          : ''
        setStatusMsg(`Patch applied: ${data.ops_applied} operations, saved as v${data.new_version}${fp}`)
        setProposals((prev) => prev.filter(p => p.id !== proposalId))

        // Reload the patched workflow onto the canvas
        if (data.patched_workflow) {
          loadWorkflowJsonRef.current?.(data.patched_workflow)
        }
      }
    } catch {
      setStatusMsg('Failed to apply patch')
    }
  }, [])

  const handleRejectPatch = useCallback(async (proposalId) => {
    try {
      // Logic for rejection (not implemented in mock but we can clear locally)
      setProposals((prev) => prev.filter(p => p.id !== proposalId))
      setStatusMsg('Patch rejected')
      setPreviewPatch(null)
    } catch {
      setStatusMsg('Failed to reject patch')
    }
  }, [])

  const handlePreviewPatch = useCallback((patch) => {
    setPreviewPatch(patch)
    const ops = JSON.parse(patch.patch_json).ops
    const affectedNodeIds = ops.filter(op => op.node_id).map(op => op.node_id)
    
    // Highlight nodes in the UI
    setNodes(nds => nds.map(n => ({
      ...n,
      className: affectedNodeIds.includes(n.id) ? 'patch-preview-highlight' : ''
    })))
  }, [setNodes])

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e) => {
      // Skip if typing in an input
      const inInput = e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT'

      // Delete / Backspace — remove selected nodes and edges
      if ((e.key === 'Delete' || e.key === 'Backspace') && !inInput) {
        const selectedNodes = nodes.filter((n) => n.selected)
        const selectedEdges = edges.filter((e) => e.selected)
        
        if (selectedNodes.length > 0 || selectedEdges.length > 0) {
          e.preventDefault()
          deleteElements({ nodes: selectedNodes, edges: selectedEdges })
          if (selectedNodes.some(n => n.id === selectedNodeId)) {
            setSelectedNodeId(null)
          }
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
  }, [handleSave, handleCompile, handlePreview, nodes, edges, deleteElements, selectedNodeId])

  const isCanvasEmpty = nodes.length === 0
  const stepGlyph = (status, busy = false) => {
    if (busy || status === 'running') return '↻'
    if (status === 'pass') return '✓'
    if (status === 'fail') return '✕'
    return '•'
  }

  return (
    <div className={`page ${embeddedMode ? 'embedded-mode' : ''}`} style={{ '--right-panel-width': `${rightPanelWidth}px` }}>
      {!embeddedMode && (
        <Palette
          components={components}
          onDragStart={() => {}}
          constraints={paletteConstraints}
        />
      )}

      <main className="canvas-wrap">
        <header className="topbar">
          <div>
            <h1>{embeddedMode ? 'Architecture Viewer' : 'Aria Designer'}</h1>
            {!embeddedMode && <p>Graph authoring workspace for user + Aria co-design</p>}
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
              <button onClick={handleSave} title="Save workflow">Save</button>
              <button onClick={handleExportJson} title="Export workflow JSON">Export</button>
              <button onClick={handleExportPython} title="Export Python module">Export Py</button>
              <button onClick={() => importInputRef.current?.click()} title="Import workflow JSON">Import JSON</button>
              <button className="primary" onClick={() => setShowImportDialog(true)} title="Import from AI Scientist">Import Research</button>
            </div>
            <div className="toolbar-group library">
              <select
                value=""
                onChange={(e) => {
                  const value = e.target.value
                  if (value) handleLoadExample(value)
                }}
              >
                <option value="">Load Example...</option>
                {exampleOptions.map((ex) => (
                  <option key={ex.value} value={ex.value}>{ex.label}</option>
                ))}
              </select>
              <button onClick={handleReloadComponents} title="Reload component library from disk">Reload</button>
              <button 
                onClick={() => setSnapToGridEnabled(!snapToGridEnabled)} 
                className={snapToGridEnabled ? 'active' : ''}
                title="Align components to grid during drag"
              >
                Snap: {snapToGridEnabled ? 'ON' : 'OFF'}
              </button>
              <button onClick={handleClearCanvas} style={{color: '#ff5050'}} title="Clear all nodes and edges from the canvas">Clear</button>
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
              <button className="primary" onClick={() => setShowAskAriaModal(true)}>Ask Aria</button>
            </div>
          </div>
        </header>

        <div className="canvas" ref={reactFlowWrapper}
          onDragEnter={() => setIsDragging(true)}
          onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget)) setIsDragging(false) }}
        >
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            defaultEdgeOptions={defaultEdgeOptions}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeDragStop={onNodeDragStop}
            isValidConnection={isValidConnection}
            onNodeClick={(_, node) => setSelectedNodeId(node.id)}
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
          {isCanvasEmpty && !isDragging && <EmptyState onLoadTemplate={handleLoadExample} />}
          <ZoomControls nodes={nodes} edges={edges} setNodes={setNodes} />
        </div>

        <footer className="statusbar">
          <span className={`run-chip run-${runStatus.phase}`}>{runStatus.phase.toUpperCase()}</span>
          <span className="status-msg">{runStatus.message}</span>
          {runStatus.metrics && Object.entries(runStatus.metrics).map(([k, v]) => (
            <span key={k} className="status-metric">{k}: {v}</span>
          ))}
          {statusMsg && <span className="status-msg">{statusMsg}</span>}
          <span>{nodes.length} nodes</span>
          <span>{edges.length} edges</span>
        </footer>
      </main>

      {!embeddedMode && (
        <aside className="panel right">
          <div 
            className={`resize-handle-left ${isResizing ? 'resizing' : ''}`} 
            onMouseDown={startResizing}
            title="Drag to resize properties panel"
          />
          <div className="panel-tabs">
            <button
              className={rightPanelTab === 'inspector' ? 'active' : ''}
              onClick={() => {
                setRightPanelTab('inspector')
                setPreviewPatch(null)
                setNodes(nds => nds.map(n => ({ ...n, className: '' })))
              }}
            >
              Properties
            </button>
            <button
              className={rightPanelTab === 'proposals' ? 'active' : ''}
              onClick={() => setRightPanelTab('proposals')}
            >
              Proposals {proposals.length > 0 ? `(${proposals.length})` : ''}
            </button>
            <button
              className={rightPanelTab === 'results' ? 'active' : ''}
              onClick={() => setRightPanelTab('results')}
            >
              Results
            </button>
          </div>

          {rightPanelTab === 'results' ? (
            <RunResultsPanel evalState={evalState} />
          ) : rightPanelTab === 'proposals' ? (
            <PatchPanel
              proposals={proposals}
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
        onClose={() => setShowAskAriaModal(false)}
        onSubmitPrompt={handleAskAriaSubmit}
        onSuggest={handleAskAriaSuggest}
        suggestions={ariaSuggestions}
        loading={ariaLoading}
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
