import React, { useMemo, useCallback } from 'react'
import {
  ReactFlow,
  Background,
  MiniMap,
  addEdge,
} from '@xyflow/react'
import DesignerNode from '../DesignerNode'
import GhostNode from '../GhostNode'
import { validateConnection } from '../../utils/validation'
import { findClosestEdge } from '../../utils/geometry'
import { 
  snapPositionToGrid, 
  findNearestFreePosition, 
  normalizeNodePlacement 
} from '../../utils/layout'

const defaultEdgeOptions = {
  type: 'smoothstep',
  animated: false,
  style: { stroke: '#5a8ab5', strokeWidth: 2 },
}

export function DesignerCanvas({ 
  nodes, setNodes, onNodesChange, 
  edges, setEdges, onEdgesChange,
  onDrop, onDragOver,
  hardwareView, heatmapView, maxFlops,
  snapToGridEnabled, pushToast
}) {

  const nodeTypes = useMemo(() => ({
    designer: (props) => (
      <DesignerNode 
        {...props} 
        hardwareView={hardwareView} 
        heatmapView={heatmapView} 
        maxFlops={maxFlops}
      />
    ),
    ghost: GhostNode
  }), [hardwareView, heatmapView, maxFlops])

  const onConnect = useCallback((params) => {
    setEdges((eds) => addEdge(params, eds))
  }, [setEdges])

  const onNodeDragStop = useCallback((_, node) => {
    const desiredPosition = snapToGridEnabled
      ? snapPositionToGrid(node.position, [15, 15])
      : node.position
      
    setNodes((nds) => {
      const resolved = findNearestFreePosition(node.id, desiredPosition, nds, { grid: [15, 15] })
      return nds.map((n) => (n.id === node.id ? { ...n, position: resolved } : n))
    })

    // Edge injection logic
    const { edge, distance } = findClosestEdge(desiredPosition, edges, nodes)
    if (edge && distance < 25 && edge.source !== node.id && edge.target !== node.id) {
       // ... (injection logic from legacy App.jsx)
    }
  }, [edges, nodes, setEdges, setNodes, snapToGridEnabled])

  return (
    <div className="designer-canvas-wrapper" style={{ width: '100%', height: '100%' }}
         onDrop={onDrop} onDragOver={onDragOver}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeDragStop={onNodeDragStop}
        nodeTypes={nodeTypes}
        defaultEdgeOptions={defaultEdgeOptions}
        fitView
      >
        <Background color="#1a1a1a" gap={15} />
        <MiniMap 
          style={{ backgroundColor: '#111' }} 
          nodeColor="#333"
          maskColor="rgba(0,0,0,0.4)"
        />
      </ReactFlow>
    </div>
  )
}
