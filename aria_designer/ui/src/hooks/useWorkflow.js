import { useCallback, useState, useRef } from 'react'
import {
  addEdge,
  useNodesState,
  useEdgesState,
  useReactFlow,
} from '@xyflow/react'
import { apiCall } from "../services/apiService"
import { buildWorkflowJson } from '../utils/workflow'
import { useHistory } from './useHistory'

export function useWorkflow(initialNodes, initialEdges, initialMeta = { name: 'Novel Architecture' }) {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)
  const [workflowMeta, setWorkflowMeta] = useState(initialMeta)
  const [saveStatus, setSaveStatus] = useState({ phase: 'idle' })
  const [runStatus, setRunStatus] = useState({ phase: 'idle', message: 'Idle' })
  const [evalState, setEvalState] = useState({ stages: [], status: null, totalTimeMs: null, error: null, benchmarking: null })
  
  const { fitView } = useReactFlow()
  const deepRunAbortRef = useRef(null)

  const { undo, redo, takeSnapshot, canUndo, canRedo } = useHistory(nodes, edges, setNodes, setEdges)

  const onConnect = useCallback((params) => {
    setEdges((eds) => addEdge(params, eds))
    takeSnapshot()
  }, [setEdges, takeSnapshot])

  const onNodeDragStop = useCallback(() => {
    takeSnapshot()
  }, [takeSnapshot])

  const handleSave = useCallback(async () => {
    setSaveStatus({ phase: 'saving' })
    try {
      const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
      const res = await apiCall(`/api/v1/workflows/${workflow.workflow_id}`, {
        method: 'PUT',
        body: workflow,
      })
      const data = await res.json()
      setSaveStatus({ phase: 'saved', version: data.version, at: new Date().toISOString() })
      setWorkflowMeta(prev => ({ ...prev, version: data.version }))
    } catch (err) {
      setSaveStatus({ phase: 'error', error: err.message })
    }
  }, [nodes, edges, workflowMeta])

  const handleValidate = useCallback(async () => {
    setRunStatus({ phase: 'validating', message: 'Validating graph...' })
    try {
      const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
      const res = await apiCall('/api/v1/workflows/validate', {
        method: 'POST',
        body: { workflow },
      })
      const data = await res.json()
      if (data.valid) {
        setRunStatus({ phase: 'validated', message: 'Graph is valid' })
      } else {
        const errorMsg = data.issues?.find(i => i.severity === 'error')?.message || 'Validation failed'
        setRunStatus({ phase: 'failed', message: `Validation Error: ${errorMsg}`, issues: data.issues })
      }
      return data.valid
    } catch (err) {
      setRunStatus({ phase: 'failed', message: `Validation failed: ${err.message}` })
      return false
    }
  }, [nodes, edges, workflowMeta])

  const handleCompile = useCallback(async () => {
    setRunStatus({ phase: 'compiling', message: 'Compiling to PyTorch...' })
    try {
      const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
      const res = await apiCall('/api/v1/workflows/compile', {
        method: 'POST',
        body: { workflow, target: 'torch' },
      })
      const data = await res.json()
      if (data.compiled) {
        setRunStatus({ phase: 'compiled', message: `Compiled successfully (${data.node_count} nodes)` })
      } else {
        setRunStatus({ phase: 'failed', message: `Compilation failed: ${data.error}` })
      }
      return data.compiled
    } catch (err) {
      setRunStatus({ phase: 'failed', message: `Compilation failed: ${err.message}` })
      return false
    }
  }, [nodes, edges, workflowMeta])

  const handleDeepRun = useCallback(async (options = {}) => {
    if (deepRunAbortRef.current) {
      deepRunAbortRef.current.abort()
    }
    
    const abortController = new AbortController()
    deepRunAbortRef.current = abortController
    
    setEvalState({ stages: [], status: 'running', totalTimeMs: null, error: null, benchmarking: null })
    setRunStatus({ phase: 'running', message: 'Starting Deep Run...' })
    
    const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
    const budget = {
      model_dim: options.model_dim || 256,
      device: options.device || 'cpu',
      run_fingerprint: true,
      run_novelty: true,
      ...options.budget
    }

    try {
      const response = await apiCall('/api/v1/workflows/evaluate/stream', {
        method: 'POST',
        body: { workflow, budget },
        signal: abortController.signal,
      })

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const payload = JSON.parse(line.substring(6))
              if (payload.stage) {
                setEvalState(prev => {
                  const stages = [...prev.stages]
                  const idx = stages.findIndex(s => s.stage === payload.stage)
                  if (idx >= 0) {
                    stages[idx] = { ...stages[idx], ...payload }
                  } else {
                    stages.push(payload)
                  }
                  return { ...prev, stages }
                })
                setRunStatus({ phase: 'running', message: `Evaluating: ${payload.stage}...` })
              } else if (payload.run_id) {
                setEvalState(prev => ({ ...prev, run_id: payload.run_id }))
              } else if (payload.status === 'success' || payload.status === 'error' || payload.status === 'failed_sandbox') {
                setEvalState(prev => ({
                  ...prev,
                  status: payload.status,
                  totalTimeMs: payload.total_time_ms,
                  error: payload.error,
                  benchmarking: payload.benchmarking || payload.result?.benchmarking
                }))
                setRunStatus({
                  phase: payload.status === 'success' ? 'success' : 'failed',
                  message: payload.status === 'success' ? 'Deep Run complete' : `Deep Run failed: ${payload.error || payload.status}`
                })
              }
            } catch (e) {
              console.warn("Malformed SSE data", e)
            }
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        setRunStatus({ phase: 'idle', message: 'Deep Run cancelled' })
      } else {
        setRunStatus({ phase: 'failed', message: `Deep Run failed: ${err.message}` })
        setEvalState(prev => ({ ...prev, status: 'error', error: err.message }))
      }
    } finally {
      deepRunAbortRef.current = null
    }
  }, [nodes, edges, workflowMeta])

  return {
    nodes, setNodes, onNodesChange,
    edges, setEdges, onEdgesChange,
    onConnect,
    workflowMeta, setWorkflowMeta,
    saveStatus, handleSave,
    runStatus, handleValidate, handleCompile, handleDeepRun,
    evalState,
    undo, redo, canUndo, canRedo, takeSnapshot,
    onNodeDragStop
  }
}
