import { useCallback, useEffect, useRef, useState } from 'react'

export function useResizablePanel({ min = 250, max = 900, initial = 300 } = {}) {
  const [width, setWidth] = useState(initial)
  const [isResizing, setIsResizing] = useState(false)
  const resizeRef = useRef({ startX: 0, startWidth: initial })

  const startResizing = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    resizeRef.current = { startX: e.clientX, startWidth: width }
    setIsResizing(true)
  }, [width])

  const handleResizeKeyDown = useCallback((e) => {
    const step = e.shiftKey ? 40 : 16
    if (e.key === 'ArrowLeft') { e.preventDefault(); setWidth((w) => Math.max(min, Math.min(max, w + step))); return }
    if (e.key === 'ArrowRight') { e.preventDefault(); setWidth((w) => Math.max(min, Math.min(max, w - step))); return }
    if (e.key === 'Home') { e.preventDefault(); setWidth(min); return }
    if (e.key === 'End') { e.preventDefault(); setWidth(max) }
  }, [min, max])

  useEffect(() => {
    if (!isResizing) return
    const handleMouseMove = (e) => {
      const deltaX = resizeRef.current.startX - e.clientX
      setWidth(Math.max(min, Math.min(max, resizeRef.current.startWidth + deltaX)))
    }
    const handleMouseUp = () => setIsResizing(false)
    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isResizing, min, max])

  return { width, isResizing, startResizing, handleResizeKeyDown }
}
