import { useCallback, useState, useRef, useEffect } from 'react'

export function useHistory(nodes, edges, setNodes, setEdges) {
  const [history, setHistory] = useState([])
  const [currentIndex, setCurrentIndex] = useState(-1)
  const isUndoRedoAction = useRef(false)

  // Take a snapshot of the current state
  const takeSnapshot = useCallback(() => {
    if (isUndoRedoAction.current) {
      isUndoRedoAction.current = false
      return
    }

    const newSnapshot = {
      nodes: JSON.parse(JSON.stringify(nodes)),
      edges: JSON.parse(JSON.stringify(edges)),
    }

    setHistory((prev) => {
      const newHistory = prev.slice(0, currentIndex + 1)
      newHistory.push(newSnapshot)
      // Limit history size to 50 steps
      if (newHistory.length > 50) {
        newHistory.shift()
        return newHistory
      }
      return newHistory
    })
    setCurrentIndex((prev) => Math.min(prev + 1, 49))
  }, [nodes, edges, currentIndex])

  const undo = useCallback(() => {
    if (currentIndex <= 0) return

    isUndoRedoAction.current = true
    const prevSnapshot = history[currentIndex - 1]
    setNodes(prevSnapshot.nodes)
    setEdges(prevSnapshot.edges)
    setCurrentIndex(currentIndex - 1)
  }, [currentIndex, history, setNodes, setEdges])

  const redo = useCallback(() => {
    if (currentIndex >= history.length - 1) return

    isUndoRedoAction.current = true
    const nextSnapshot = history[currentIndex + 1]
    setNodes(nextSnapshot.nodes)
    setEdges(nextSnapshot.edges)
    setCurrentIndex(currentIndex + 1)
  }, [currentIndex, history, setNodes, setEdges])

  // Initial snapshot
  useEffect(() => {
    if (history.length === 0 && nodes.length > 0) {
      takeSnapshot()
    }
  }, [nodes, takeSnapshot, history.length])

  return {
    undo,
    redo,
    takeSnapshot,
    canUndo: currentIndex > 0,
    canRedo: currentIndex < history.length - 1
  }
}
