import React, { useMemo } from 'react'
import {
  ReactFlow,
  Background,
  MiniMap,
} from '@xyflow/react'
import DesignerNode from '../DesignerNode'
import GhostNode from '../GhostNode'

const defaultEdgeOptions = {
  type: 'smoothstep',
  animated: false,
  style: { stroke: '#5a8ab5', strokeWidth: 2 },
}

export function DesignerCanvas({
  nodes, setNodes, onNodesChange,
  edges, setEdges, onEdgesChange,
  onConnect, onNodeDragStop,
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
