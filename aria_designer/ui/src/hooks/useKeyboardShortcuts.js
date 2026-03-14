import { useEffect } from 'react'

/**
 * Global keyboard shortcut listener for the designer.
 * @param {Object} handlers - named callbacks: handleSave, handleCompile, handlePreview,
 *   handleDeleteSelection, undoGraph, redoGraph, setShowShortcuts, setSelectedNodeId,
 *   nodes, setNodes, snapToGridEnabled
 */
export function useKeyboardShortcuts({
  handleSave,
  handleCompile,
  handlePreview,
  handleDeleteSelection,
  undoGraph,
  redoGraph,
  setShowShortcuts,
  setSelectedNodeId,
  nodes,
  setNodes,
  snapToGridEnabled,
}) {
  useEffect(() => {
    const handler = (e) => {
      const inInput = e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT'

      // Delete / Backspace — remove selected nodes and edges
      if ((e.key === 'Delete' || e.key === 'Backspace') && !inInput && !e.target.isContentEditable) {
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
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [handleSave, handleCompile, handlePreview, handleDeleteSelection, nodes, setNodes, snapToGridEnabled, undoGraph, redoGraph, setShowShortcuts, setSelectedNodeId])
}
