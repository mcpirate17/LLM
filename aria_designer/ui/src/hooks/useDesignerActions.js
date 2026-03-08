import { useCallback, useState, useRef } from 'react'
import { apiCall } from '../services/apiService'
import { buildWorkflowJson } from '../utils/workflow'

export function useDesignerActions(state) {
  const {
    nodes, setNodes, edges, setEdges,
    workflowMeta, setWorkflowMeta,
    setStatusMsg, pushToast,
    setRightPanelTab, setWorkflowStage, setStepStatus,
    setRunStatus, setEvalState, setAllNodeEvalStatus,
    clearNodeHighlights, highlightNodeErrors,
    loadWorkflowJson, benchmarkObserved
  } = state

  const [saveState, setSaveState] = useState({ phase: 'idle', message: '', version: null, fingerprint: null, at: 0 })
  const [validateUi, setValidateUi] = useState({ inProgress: false, last: 'idle', issues: 0 })
  const deepRunAbortRef = useRef(null)

  const handleValidate = useCallback(async () => {
    const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
    setWorkflowStage('validate')
    setStepStatus((s) => ({ ...s, validate: 'running' }))
    setValidateUi({ inProgress: true, last: 'idle', issues: 0 })
    setStatusMsg('Validating workflow...')
    setRunStatus({ phase: 'running', message: 'Validation in progress...', metrics: null })
    clearNodeHighlights()
    try {
      const res = await apiCall(`/api/v1/workflows/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflow }),
      })
      const data = await res.json()
      const hasErrors = (data.issues || []).some((i) => i.severity === 'error')
      
      if (data.success === true || data.valid === true || (data.issues && !hasErrors)) {
        setStepStatus((s) => ({ ...s, validate: 'pass' }))
        setValidateUi({ inProgress: false, last: 'pass', issues: 0 })
        setStatusMsg('Validation passed')
        setRunStatus({ phase: 'success', message: 'Validation passed', metrics: { issues: 0 } })
        return true
      }
      setStepStatus((s) => ({ ...s, validate: 'fail' }))
      setValidateUi({ inProgress: false, last: 'fail', issues: (data.issues || []).length })
      setStatusMsg('Validation failed')
      return false
    } catch (err) {
      setStepStatus((s) => ({ ...s, validate: 'fail' }))
      setStatusMsg('Validation failed (API error)')
      return false
    }
  }, [nodes, edges, workflowMeta, clearNodeHighlights, setStatusMsg, setRunStatus, setStepStatus, setWorkflowStage])

  const handleDeepRun = useCallback(async () => {
    if (deepRunAbortRef.current) deepRunAbortRef.current.abort()
    const controller = new AbortController()
    deepRunAbortRef.current = controller

    const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
    setEvalState({ stages: [], status: 'running', totalTimeMs: null, error: null, benchmarking: null })
    setRightPanelTab('results')
    setStatusMsg('Run: starting...')
    
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
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const payload = JSON.parse(line.slice(6))
            if (payload.stage) {
               setEvalState(prev => ({
                 ...prev,
                 stages: [...prev.stages.filter(s => s.stage !== payload.stage), payload]
               }))
            }
            if (payload.status === 'success' || payload.status === 'error') {
               setStatusMsg(payload.status === 'success' ? 'Run complete' : `Run failed: ${payload.error}`)
            }
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setStatusMsg(`Run failed: ${err.message}`)
      }
    }
  }, [nodes, edges, workflowMeta, setEvalState, setRightPanelTab, setStatusMsg])

  const handleSave = useCallback(async () => {
    const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
    const newId = `wf_${Date.now().toString(36)}`
    setSaveState({ phase: 'saving', message: 'Saving…', version: null, fingerprint: null, at: Date.now() })
    try {
      const res = await apiCall(`/api/v1/workflows/${newId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(workflow),
      })
      const data = await res.json()
      setWorkflowMeta((prev) => ({ ...prev, workflow_id: newId, metadata: { ...prev.metadata, graph_fingerprint: data.fingerprint } }))
      setSaveState({ phase: 'saved', message: 'Saved', at: Date.now() })
      setStatusMsg(`Saved as ${newId}`)
    } catch (err) {
      setStatusMsg(`Save failed: ${err.message}`)
      setSaveState({ phase: 'failed', message: 'Save failed', at: Date.now() })
    }
  }, [nodes, edges, workflowMeta, setWorkflowMeta, setStatusMsg])

  return {
    handleValidate,
    handleSave,
    handleDeepRun,
    saveState,
    validateUi,
  }
}
