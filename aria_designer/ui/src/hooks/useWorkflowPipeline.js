import { useCallback, useEffect, useRef, useState } from 'react'
import { apiCall } from '../services/apiService'
import { buildWorkflowJson } from '../utils/workflow'

/**
 * Workflow pipeline: validate → compile → preview → deep run.
 * Manages evalState, stepStatus, validateUi, and SSE streaming.
 */
export function useWorkflowPipeline({
  nodes, edges, setNodes, workflowMeta,
  clearNodeHighlights, highlightNodeErrors, collectFailureErrorMap, setAllNodeEvalStatus,
  setStatusMsg, setRightPanelTab,
  importedBaseline, benchmarkObserved,
}) {
  const [workflowStage, setWorkflowStage] = useState('idle')
  const [evalState, setEvalState] = useState({
    runId: null,
    workflowId: workflowMeta?.workflow_id || null,
    stages: [],
    status: null,
    totalTimeMs: null,
    error: null,
    benchmarking: null,
    semanticWarnings: [],
    errorDetails: null,
    discoveryUrl: null,
  })
  const [validateUi, setValidateUi] = useState({ inProgress: false, last: 'idle', issues: 0 })
  const [stepStatus, setStepStatus] = useState({
    validate: 'idle',
    compile: 'idle',
    test: 'idle',
    run: 'idle',
  })
  const [runStatus, setRunStatus] = useState({ phase: 'idle', message: 'Idle', metrics: null })
  const deepRunAbortRef = useRef(null)
  const validateTimersRef = useRef({})

  // Cleanup validate timers on unmount
  useEffect(() => () => {
    Object.values(validateTimersRef.current).forEach((t) => clearTimeout(t))
  }, [])

  const validateNodeConfig = useCallback((nodeId, componentId, config) => {
    const timerKey = `${nodeId}_${componentId}`
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
      if (!res.ok) {
        if (res.status !== 400 && res.status !== 422) {
          throw new Error(`validate ${res.status}`)
        }
      }
      const data = await res.json()
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
  }, [nodes, edges, workflowMeta, clearNodeHighlights, highlightNodeErrors, setStatusMsg])

  const handleCompile = useCallback(async () => {
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
      const res = await apiCall(`/api/v1/workflows/compile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflow, target: 'cpu' }),
      })
      if (!res.ok) throw new Error(`compile ${res.status}`)
      const data = await res.json()

      const compiled = data.success === true || data.compiled === true
      if (compiled) {
        setStepStatus((s) => ({ ...s, compile: 'pass' }))
        setStatusMsg(`Compiled successfully: ${data.n_ops || 0} ops, ${data.param_count || 0} params`)
        setRunStatus({
          phase: 'success',
          message: 'Compile succeeded',
          metrics: { params: data.param_count || 0, ops: data.n_ops || 0, depth: data.depth || 0 },
        })
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
          `${guidance ? ` \u2014 ${guidance}` : ''}` +
          `${warning ? ` \u2014 warning: ${warning}` : ''}`
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
  }, [nodes, edges, workflowMeta, stepStatus, handleValidate, clearNodeHighlights, highlightNodeErrors, collectFailureErrorMap, setStatusMsg])

  const handlePreview = useCallback(async () => {
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

      const res = await apiCall(`/api/v1/workflows/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflow }),
      })
      if (!res.ok) throw new Error(`preview ${res.status}`)
      const data = await res.json()

      if (data.success === true) {
        setStepStatus((s) => ({ ...s, test: 'pass' }))
        if (data.metrics) {
          setStatusMsg(
            `Run complete \u2014 params ${data.metrics.param_count || 0}, FLOPs/token ${data.metrics.flops_per_token || 0}, ` +
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
          setStatusMsg(`Run complete \u2014 ${nResults} output${nResults !== 1 ? 's' : ''} computed`)
          setRunStatus({
            phase: 'success',
            message: 'Run succeeded',
            metrics: { outputs: nResults },
          })
          setNodes(nds => nds.map(n => {
            if (data.results[n.id]) {
              return { ...n, data: { ...n.data, preview: data.results[n.id] } }
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
  }, [nodes, edges, workflowMeta, stepStatus, handleCompile, setNodes, clearNodeHighlights, highlightNodeErrors, collectFailureErrorMap, setStatusMsg])

  const handleDeepRun = useCallback(async () => {
    if (stepStatus.compile !== 'pass') {
      const compiled = await handleCompile()
      if (!compiled) {
        setStatusMsg('Run skipped: fix validation/compile errors first')
        return
      }
    }
    if (deepRunAbortRef.current) deepRunAbortRef.current.abort()
    const controller = new AbortController()
    deepRunAbortRef.current = controller
    setWorkflowStage('deep-run')
    setStepStatus((s) => ({ ...s, run: 'running' }))

    const workflow = buildWorkflowJson(nodes, edges, workflowMeta)
    setEvalState({
      runId: null,
      workflowId: workflowMeta?.workflow_id || null,
      stages: [],
      status: 'running',
      totalTimeMs: null,
      error: null,
      benchmarking: null,
      semanticWarnings: [],
      errorDetails: null,
      discoveryUrl: null,
    })
    setRightPanelTab('results')
    setStatusMsg('Run: starting...')
    setRunStatus({ phase: 'running', message: 'Run: starting...', metrics: null })
    setAllNodeEvalStatus('running', null)

    try {
      const budget = { run_fingerprint: true, run_novelty: true }
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

        const lines = buffer.split('\n')
        buffer = lines.pop()

        let eventType = null
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim()
          } else if (line.startsWith('data: ') && eventType) {
            try {
              const payload = JSON.parse(line.slice(6))
              processSSEEvent(eventType, payload)
            } catch {
              // ignore malformed JSON
            }
            eventType = null
          }
        }
      }

      // Process remaining buffer
      if (buffer.trim()) {
        const remaining = buffer.split('\n')
        let et = null
        for (const line of remaining) {
          if (line.startsWith('event: ')) {
            et = line.slice(7).trim()
          } else if (line.startsWith('data: ') && et) {
            try {
              const payload = JSON.parse(line.slice(6))
              processSSEEvent(et, payload)
            } catch {
              // ignore malformed JSON
            }
            et = null
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
        setEvalState((prev) => ({
          ...prev,
          status: 'error',
          error: err.message,
          errorDetails: {
            stage: 'transport',
            error_type: 'transport_error',
            error_message: err.message,
            root_cause_code: 'transport_error',
          },
        }))
        setRunStatus({ phase: 'failed', message: `Run failed: ${err.message}`, metrics: null })
        setAllNodeEvalStatus('fail', err.message)
      }
    }

    function processSSEEvent(eventType, payload) {
      if (eventType === 'run_id') {
        setEvalState((prev) => ({ ...prev, runId: payload.run_id }))
      } else if (eventType === 'semantic_warnings') {
        setEvalState((prev) => ({
          ...prev,
          semanticWarnings: Array.isArray(payload.warnings) ? payload.warnings : [],
        }))
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

        if (payload.status === 'running') {
          setRunStatus({ phase: 'running', message: `Run: ${payload.stage}...`, metrics: null })
        }

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

        if (payload.stage === 'profiling' && payload.status === 'done' && payload.metrics?.op_profiles) {
          const profileMap = {}
          for (const op of payload.metrics.op_profiles) {
            if (op.aria_node_id) profileMap[op.aria_node_id] = op
          }
          setNodes((nds) =>
            nds.map((n) =>
              profileMap[n.id] ? { ...n, data: { ...n.data, profile: profileMap[n.id] } } : n
            )
          )
        }

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

        if (payload.stage === 'sandbox' && payload.status === 'done' && payload.metrics && !payload.metrics.passed) {
          setAllNodeEvalStatus('fail', 'Sandbox evaluation failed')
        }
      } else if (eventType === 'done') {
        const succeeded = payload.status === 'success'
        setStepStatus((s) => ({ ...s, run: succeeded ? 'pass' : 'fail' }))
        const benchmarking = payload.benchmarking || payload.result?.benchmarking || null
        const summary = benchmarking?.summary || null
        const compositeScore = payload.composite_score ?? payload.result?.composite_score ?? null
        const graphFingerprint = payload.graph_fingerprint ?? payload.result?.graph_fingerprint ?? null
        const scoreText = compositeScore != null
          ? `, score ${Number(compositeScore).toFixed(2)}`
          : summary?.score != null ? `, score ${Number(summary.score).toFixed(2)}` : ''
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
          compositeScore: compositeScore ?? prev.compositeScore ?? null,
          graphFingerprint: graphFingerprint ?? prev.graphFingerprint ?? null,
          discoveryUrl: payload.discovery_url ?? payload.result?.discovery_url ?? prev.discoveryUrl ?? null,
          semanticWarnings: payload.result?.semantic_warnings ?? prev.semanticWarnings ?? [],
          errorDetails: payload.error_details ?? payload.result?.error_details ?? prev.errorDetails ?? null,
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
        } else if (payload.error_stage) {
          setNodes((nds) =>
            nds.map((n) => ({
              ...n,
              data: {
                ...n.data,
                evalStatus: 'fail',
                evalError: n.data.evalError || payload.error || null,
              },
            }))
          )
        }
      }
    }
  }, [nodes, edges, workflowMeta, stepStatus, handleCompile, setNodes, setAllNodeEvalStatus, importedBaseline, benchmarkObserved, clearNodeHighlights, highlightNodeErrors, collectFailureErrorMap, setStatusMsg, setRightPanelTab])

  return {
    workflowStage, setWorkflowStage,
    evalState, setEvalState,
    validateUi,
    stepStatus,
    runStatus, setRunStatus,
    handleValidate,
    handleCompile,
    handlePreview,
    handleDeepRun,
    onParamChange,
  }
}
