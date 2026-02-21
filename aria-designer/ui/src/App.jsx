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
import { isValidConnection as validateConnection } from './utils/validation'
import { buildWorkflowJson } from './utils/workflow'
import { starterEdges, starterNodes } from './mockData'

const API_BASE = 'http://127.0.0.1:8091'
const DESIGNER_API_BASE = import.meta.env.VITE_DESIGNER_API_BASE || 'http://127.0.0.1:5000'

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
  const [ariaSuggestions, setAriaSuggestions] = useState([])
  const [ariaLoading, setAriaLoading] = useState(false)

  const [showShortcuts, setShowShortcuts] = useState(false)
  const [showImportDialog, setShowImportDialog] = useState(false)
  const [helpRequest, setHelpRequest] = useState(null)
  const [evalState, setEvalState] = useState({ stages: [], status: null, totalTimeMs: null, error: null })
  const deepRunAbortRef = useRef(null)
  const reactFlowWrapper = useRef(null)
  const importInputRef = useRef(null)
  const { screenToFlowPosition, deleteElements } = useReactFlow()

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
    fetch(`${API_BASE}/api/v1/components?status=approved`)
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
        const r = await fetch(`${API_BASE}/api/v1/aria/proposals?status=pending`)
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

  // Autosave local draft on canvas changes.
  useEffect(() => {
    const workflow = buildWorkflowJson(nodes, edges)
    localStorage.setItem('aria-workflow-autosave', JSON.stringify(workflow))
  }, [nodes, edges])

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
  ]), [])

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge({ ...params }, eds)),
    [setEdges]
  )

  // Drag-and-drop from palette onto canvas
  const onDragOver = useCallback((e) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop = useCallback(
    (e) => {
      e.preventDefault()
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
          performance: comp.performance || {},
          manifest: comp,
        },
      }
      setNodes((prev) => [...prev, newNode])
      setSelectedNodeId(newId)
    },
    [screenToFlowPosition, setNodes]
  )

  // Click palette item to add (fallback for non-drag)
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
          performance: comp.performance || {},
          manifest: comp,
        },
      }
      setNodes((prev) => [...prev, newNode])
      setSelectedNodeId(newId)
    },
    [setNodes]
  )

  // Update node param
  const onParamChange = useCallback(
    (nodeId, paramName, value) => {
      setNodes((prev) =>
        prev.map((n) =>
          n.id === nodeId
            ? { ...n, data: { ...n.data, paramValues: { ...n.data.paramValues, [paramName]: value } } }
            : n
        )
      )
    },
    [setNodes]
  )

  const [isRunDisabled, setIsRunDisabled] = useState(false)
  
  // Real-time connection validation
  const isValidConnection = useCallback((connection) => {
    return validateConnection(connection, nodes)
  }, [nodes])

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
    const workflow = buildWorkflowJson(nodes, edges)
    setWorkflowStage('validate')
    clearNodeHighlights()
    try {
      let data
      try {
        const res = await fetch(`${DESIGNER_API_BASE}/api/designer/validate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(workflow),
        })
        data = await res.json()
      } catch {
        const res = await fetch(`${API_BASE}/api/v1/workflows/validate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ workflow }),
        })
        data = await res.json()
      }

      if (data.success === true || data.valid === true) {
        setStatusMsg('Valid graph')
        setRunStatus({ phase: 'idle', message: 'Validation passed', metrics: null })
        return
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
      setStatusMsg(messages.length > 0 ? `Validation failed: ${messages.join('; ')}` : 'Validation failed')
      setRunStatus({ phase: 'failed', message: 'Validation failed', metrics: null })
    } catch {
      setStatusMsg('Validation failed (API offline)')
      setRunStatus({ phase: 'failed', message: 'Validation failed (API offline)', metrics: null })
    }
  }, [nodes, edges, clearNodeHighlights, highlightNodeErrors])

  const handleCompile = useCallback(async () => {
    const workflow = buildWorkflowJson(nodes, edges)
    setWorkflowStage('compile')
    clearNodeHighlights()
    setRunStatus({ phase: 'compiling', message: 'Compiling workflow...', metrics: null })
    try {
      let data
      let usedDesignerApi = true
      try {
        const res = await fetch(`${DESIGNER_API_BASE}/api/designer/compile`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(workflow),
        })
        data = await res.json()
      } catch {
        usedDesignerApi = false
        const res = await fetch(`${API_BASE}/api/v1/workflows/compile`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ workflow, target: 'cpu' }),
        })
        data = await res.json()
      }

      const compiled = data.success === true || data.compiled === true
      if (compiled) {
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
      } else {
        const err = data.error || 'Compilation failed'
        const nodeMatch = String(err).match(/node\\s+([a-zA-Z0-9_:-]+)/i)
        if (nodeMatch && nodeMatch[1]) {
          highlightNodeErrors({ [nodeMatch[1]]: [err] })
        }
        setStatusMsg(`Compilation failed: ${err}`)
        setRunStatus({ phase: 'failed', message: `Compile failed: ${err}`, metrics: null })
      }
    } catch {
      setStatusMsg('Compilation failed (API offline)')
      setRunStatus({ phase: 'failed', message: 'Compile failed (API offline)', metrics: null })
    }
  }, [nodes, edges, clearNodeHighlights, highlightNodeErrors])

  const handleSave = useCallback(async () => {
    const workflow = buildWorkflowJson(nodes, edges)
    try {
      try {
        const res = await fetch(`${DESIGNER_API_BASE}/api/designer/save`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(workflow),
        })
        const data = await res.json()
        if (!data.success) throw new Error(data.error || 'Save failed')
        setStatusMsg('Workflow saved')
      } catch {
        await fetch(`${API_BASE}/api/v1/workflows/${workflow.workflow_id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(workflow),
        })
        setStatusMsg('Workflow saved')
      }
    } catch {
      // Save to localStorage as fallback
      localStorage.setItem('aria-workflow', JSON.stringify(workflow))
      setStatusMsg('Saved to browser (API offline)')
    }
  }, [nodes, edges])

  const handlePreview = useCallback(async () => {
    const workflow = buildWorkflowJson(nodes, edges)
    try {
      setWorkflowStage('run')
      clearNodeHighlights()
      setStatusMsg('Running preview...')
      setRunStatus({ phase: 'running', message: 'Running forward pass...', metrics: null })
      let data
      let usedDesignerApi = true
      try {
        const res = await fetch(`${DESIGNER_API_BASE}/api/designer/run`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(workflow),
        })
        data = await res.json()
      } catch {
        usedDesignerApi = false
        const res = await fetch(`${API_BASE}/api/v1/workflows/preview`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ workflow }),
        })
        data = await res.json()
      }

      if (data.success === true) {
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
        const err = data.error || 'Run failed'
        const nodeMatch = String(err).match(/node\\s+([a-zA-Z0-9_:-]+)/i)
        if (nodeMatch && nodeMatch[1]) {
          highlightNodeErrors({ [nodeMatch[1]]: [err] })
        }
        setStatusMsg(`Run failed: ${err}`)
        setRunStatus({ phase: 'failed', message: `Run failed: ${err}`, metrics: null })
      }
    } catch {
      setStatusMsg('Preview failed (API offline)')
      setRunStatus({ phase: 'failed', message: 'Run failed (API offline)', metrics: null })
    }
  }, [nodes, edges, setNodes, clearNodeHighlights, highlightNodeErrors])

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
    // Abort any in-flight deep run
    if (deepRunAbortRef.current) deepRunAbortRef.current.abort()
    const controller = new AbortController()
    deepRunAbortRef.current = controller

    const workflow = buildWorkflowJson(nodes, edges)
    setEvalState({ stages: [], status: 'running', totalTimeMs: null, error: null })
    setRightPanelTab('results')
    setRunStatus({ phase: 'running', message: 'Deep Run: starting...', metrics: null })

    // Mark all nodes as running
    setAllNodeEvalStatus('running', null)

    try {
      const res = await fetch(`${API_BASE}/api/v1/workflows/evaluate/stream`, {
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

                // Sandbox failure: mark all nodes as failed with the sandbox error
                if (payload.stage === 'sandbox' && payload.status === 'done' && payload.metrics && !payload.metrics.passed) {
                  setAllNodeEvalStatus('fail', 'Sandbox evaluation failed')
                }
              } else if (eventType === 'done') {
                const succeeded = payload.status === 'success'
                setEvalState((prev) => ({
                  ...prev,
                  status: payload.status,
                  totalTimeMs: payload.total_time_ms,
                  error: payload.error || null,
                }))
                setRunStatus({
                  phase: succeeded ? 'success' : 'failed',
                  message: succeeded
                    ? `Deep Run complete (${(payload.total_time_ms / 1000).toFixed(1)}s)`
                    : `Deep Run failed: ${payload.error || payload.status}`,
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
        setRunStatus({ phase: 'idle', message: 'Deep Run cancelled', metrics: null })
        setAllNodeEvalStatus(null, null)
      } else {
        setEvalState((prev) => ({ ...prev, status: 'error', error: err.message }))
        setRunStatus({ phase: 'failed', message: `Deep Run failed: ${err.message}`, metrics: null })
        setAllNodeEvalStatus('fail', err.message)
      }
    }
  }, [nodes, edges, setNodes, setAllNodeEvalStatus])

  const handleExportJson = useCallback(() => {
    const workflow = buildWorkflowJson(nodes, edges)
    const blob = new Blob([JSON.stringify(workflow, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `workflow_${workflow.workflow_id}.json`
    a.click()
    URL.revokeObjectURL(url)
    setStatusMsg('Exported JSON')
  }, [nodes, edges])

  const handleExportPython = useCallback(async () => {
    const workflow = buildWorkflowJson(nodes, edges)
    try {
      const res = await fetch(`${DESIGNER_API_BASE}/api/designer/export/python`, {
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
  }, [nodes, edges])

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
      })
    })
  }, [components, setEdges, setNodes])

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
      const res = await fetch(path)
      const data = await res.json()
      loadWorkflowJson(data)
    } catch {
      setStatusMsg('Failed to load example')
    }
  }, [loadWorkflowJson])

  const handleReloadComponents = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/components/reload`, { method: 'POST' })
      const data = await res.json()
      const list = await fetch(`${API_BASE}/api/v1/components?status=approved`)
      const comps = await list.json()
      setComponents(comps)
      setStatusMsg(`Reloaded components: ${data.reloaded || comps.length}`)
    } catch {
      setStatusMsg('Failed to reload components (API offline)')
    }
  }, [])

  const handleAskAriaSuggest = useCallback(async () => {
    const workflow = buildWorkflowJson(nodes, edges)
    setAriaLoading(true)
    try {
      const res = await fetch(`${API_BASE}/api/v1/aria/suggest-components`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflow }),
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
  }, [nodes, edges])

  const handleAskAriaSubmit = useCallback(async (promptText) => {
    const prompt = String(promptText || '').trim()
    if (!prompt) return
    const workflow = buildWorkflowJson(nodes, edges)
    setAriaLoading(true)

    // Preferred path: backend-generated deterministic patch proposal.
    try {
      const res = await fetch(`${API_BASE}/api/v1/aria/generate-patch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflow, prompt, base_version: 1 }),
      })
      const data = await res.json()
      if (res.ok && data.proposal_id) {
        setStatusMsg(`Aria proposal created: ${data.proposal_id}`)
        setRightPanelTab('proposals')
        setShowAskAriaModal(false)
        const pRes = await fetch(`${API_BASE}/api/v1/aria/proposals?status=pending`)
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
        rationale: `Prompt: ${prompt}`,
        expected_impact: { summary: 'User-directed change proposal' },
        ops,
      }
      const res = await fetch(`${API_BASE}/api/v1/aria/propose-patch`, {
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
      const pRes = await fetch(`${API_BASE}/api/v1/aria/proposals?status=pending`)
      const pData = await pRes.json()
      setProposals(Array.isArray(pData) ? pData : [])
    } catch (err) {
      setStatusMsg(`Failed to create proposal: ${err.message || err}`)
    } finally {
      setAriaLoading(false)
    }
  }, [nodes, edges, ariaSuggestions])

  // Patch Application
  const handleApplyPatch = useCallback(async (proposalId) => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/aria/apply-patch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ proposal_id: proposalId, approved_by: 'user' }),
      })
      const data = await res.json()
      if (data.applied) {
        setStatusMsg(`Patch applied: ${data.ops_applied} operations.`)
        // Clear proposal and refetch
        setProposals((prev) => prev.filter(p => p.id !== proposalId))
        
        // In a real system, we would apply the patch transformation locally 
        // OR refetch the whole workflow. For now, we refetch proposals.
        // The mock backend doesn't actually mutate the graph state in DB yet.
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

      // Delete / Backspace — remove selected nodes
      if ((e.key === 'Delete' || e.key === 'Backspace') && !inInput) {
        const selected = nodes.filter((n) => n.selected)
        if (selected.length > 0) {
          e.preventDefault()
          deleteElements({ nodes: selected })
          setSelectedNodeId(null)
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
  }, [handleSave, handleCompile, handlePreview, nodes, deleteElements])

  const isCanvasEmpty = nodes.length === 0

  return (
    <div className="page">
      <Palette components={components} onDragStart={() => {}} />

      <main className="canvas-wrap">
        <header className="topbar">
          <div>
            <h1>Aria Designer</h1>
            <p>Graph authoring workspace for user + Aria co-design</p>
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
              <button className={workflowStage === 'validate' ? 'active' : ''} onClick={handleValidate}>Validate</button>
              <button className={workflowStage === 'compile' ? 'active' : ''} onClick={handleCompile}>Compile</button>
              <button 
                className={`primary ${workflowStage === 'run' ? 'active' : ''}`} 
                onClick={handlePreview}
                disabled={isRunDisabled}
                title={isRunDisabled ? "Fix validation errors to run" : "Run forward pass"}
              >
                Run
              </button>
              <button
                className={evalState.status === 'running' ? 'active' : ''}
                onClick={handleDeepRun}
                disabled={isRunDisabled}
                title="Full pipeline: profile + sandbox + fingerprint + novelty"
              >
                Deep Run
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
              <button onClick={handleReloadComponents}>Reload</button>
            </div>
            <div className="toolbar-group ai">
              <button className="primary" onClick={() => setShowAskAriaModal(true)}>Ask Aria</button>
            </div>
          </div>
        </header>

        <div className="canvas" ref={reactFlowWrapper}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            defaultEdgeOptions={defaultEdgeOptions}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            isValidConnection={isValidConnection}
            onNodeClick={(_, node) => setSelectedNodeId(node.id)}
            onPaneClick={() => setSelectedNodeId(null)}
            onDragOver={onDragOver}
            onDrop={onDrop}
            fitView
            snapToGrid
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
          {isCanvasEmpty && <EmptyState onLoadTemplate={handleLoadExample} />}
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

      <aside className="panel right">
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
          <Inspector
            selectedNode={selectedNode}
            allComponents={components}
            nodeCount={nodes.length}
            edgeCount={edges.length}
            onParamChange={onParamChange}
            helpRequest={helpRequest}
          />
        )}
      </aside>

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
