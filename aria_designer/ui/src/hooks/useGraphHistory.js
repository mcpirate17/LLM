import { useCallback, useRef, useState } from 'react'

/**
 * Undo/redo graph history with signature-based deduplication.
 * @param {Function} setNodes - React Flow setNodes
 * @param {Function} setEdges - React Flow setEdges
 * @param {Function} pushToast - toast notification callback
 * @returns {{ captureSnapshot, undoGraph, redoGraph, historyUi, skipHistoryRef }}
 */
export function useGraphHistory(setNodes, setEdges, pushToast) {
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

  return { captureSnapshot, undoGraph, redoGraph, historyUi, skipHistoryRef, historyRef }
}
