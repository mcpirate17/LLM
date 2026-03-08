import { useCallback } from 'react'
import { addEdge, useReactFlow } from '@xyflow/react'
import { findClosestEdge, validateConnection } from '../utils/geometry'
import { snapPositionToGrid, findNearestFreePosition } from '../utils/layout'

export function useDesignerCanvasEvents(state) {
  const {
    nodes, setNodes, edges, setEdges,
    snapToGridEnabled, setStatusMsg, pushToast,
    setSelectedNodeId, setIsDragging, paletteConstraints
  } = state

  const { screenToFlowPosition, getViewport } = useReactFlow()

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge({ ...params }, eds)),
    [setEdges]
  )

  const onNodeDragStop = useCallback(
    (_, node) => {
      const desiredPosition = snapToGridEnabled
        ? snapPositionToGrid(node.position, [15, 15])
        : node.position
      setNodes((nds) => {
        const resolved = findNearestFreePosition(node.id, desiredPosition, nds, { grid: [15, 15] })
        return nds.map((n) => (n.id === node.id ? { ...n, position: resolved } : n))
      })

      // Auto-injection logic
      const { edge, distance } = findClosestEdge(desiredPosition, edges, nodes)
      if (edge && distance < 25 && edge.source !== node.id && edge.target !== node.id) {
        const c1 = {
          source: edge.source,
          sourceHandle: edge.sourceHandle,
          target: node.id,
          targetHandle: node.data.inputs[0]?.name || 'x',
        }
        const c2 = {
          source: node.id,
          sourceHandle: node.data.outputs[0]?.name || 'y',
          target: edge.target,
          targetHandle: edge.targetHandle,
        }
        const filteredPreview = edges.filter((e) => e.id !== edge.id)
        if (validateConnection(c1, nodes, filteredPreview) && 
            validateConnection(c2, nodes, [...filteredPreview, { id: 'tmp_inject_1', ...c1 }])) {
          setEdges((eds) => {
            const filtered = eds.filter((e) => e.id !== edge.id)
            return [
              ...filtered,
              { id: `e_inject_1_${Date.now()}`, ...c1 },
              { id: `e_inject_2_${Date.now()}`, ...c2 },
            ]
          })
        }
      }
      state.setDragGuides({ x: null, y: null })
    },
    [edges, nodes, setEdges, setNodes, snapToGridEnabled, pushToast, state]
  )

  const onDrop = useCallback(
    (e) => {
      e.preventDefault()
      setIsDragging(false)
      const raw = e.dataTransfer.getData('application/aria-component')
      if (!raw) return

      const comp = JSON.parse(raw)
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY })
      const desiredPos = snapToGridEnabled ? snapPositionToGrid(position, [15, 15]) : position
      
      const newId = `node_${Date.now()}`
      const newNode = {
        id: newId,
        type: 'designer',
        position: desiredPos,
        data: {
          label: comp.name || comp.id,
          category: comp.category,
          componentId: comp.id,
          inputs: comp.inputs || [],
          outputs: comp.outputs || [],
          params: comp.params || {},
          manifest: comp,
        },
      }

      setNodes((prev) => [...prev, newNode])
      setSelectedNodeId(newId)
    },
    [screenToFlowPosition, setNodes, snapToGridEnabled, setSelectedNodeId, setIsDragging]
  )

  const onDragOver = useCallback((e) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }, [])

  return {
    onConnect,
    onNodeDragStop,
    onDrop,
    onDragOver
  }
}
