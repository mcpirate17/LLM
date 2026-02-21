import { useCallback, useEffect, useState } from 'react'
import { useReactFlow, useViewport } from '@xyflow/react'
import dagre from '@dagrejs/dagre'
import { ZoomIn, ZoomOut, Maximize, Square, LayoutGrid } from 'lucide-react'

export default function ZoomControls({ nodes, edges, setNodes }) {
  const { zoomIn, zoomOut, fitView, zoomTo } = useReactFlow()
  const { zoom } = useViewport()
  const [zoomPct, setZoomPct] = useState(100)

  useEffect(() => {
    setZoomPct(Math.round(zoom * 100))
  }, [zoom])

  const handleAutoLayout = useCallback(() => {
    if (!nodes || nodes.length === 0) return

    const g = new dagre.graphlib.Graph()
    g.setDefaultEdgeLabel(() => ({}))
    g.setGraph({ rankdir: 'TB', nodesep: 60, ranksep: 80 })

    nodes.forEach((node) => {
      g.setNode(node.id, { width: 160, height: 80 })
    })

    edges.forEach((edge) => {
      g.setEdge(edge.source, edge.target)
    })

    dagre.layout(g)

    setNodes((nds) =>
      nds.map((node) => {
        const pos = g.node(node.id)
        if (!pos) return node
        return {
          ...node,
          position: { x: pos.x - 80, y: pos.y - 40 },
        }
      })
    )

    setTimeout(() => fitView({ padding: 0.15 }), 50)
  }, [nodes, edges, setNodes, fitView])

  return (
    <div className="zoom-controls">
      <button onClick={() => zoomIn()} title="Zoom In"><ZoomIn size={16} /></button>
      <button onClick={() => zoomOut()} title="Zoom Out"><ZoomOut size={16} /></button>
      <button onClick={() => fitView({ padding: 0.15 })} title="Fit to View"><Maximize size={16} /></button>
      <button onClick={() => zoomTo(1)} title="Reset to 100%"><Square size={16} /></button>
      <button onClick={handleAutoLayout} title="Auto Layout (DAG)"><LayoutGrid size={16} /></button>
      <span className="zoom-label">{zoomPct}%</span>
    </div>
  )
}
