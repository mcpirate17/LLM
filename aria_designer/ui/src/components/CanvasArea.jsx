import { memo, useCallback } from 'react'
import {
  Background,
  MiniMap,
  ReactFlow,
} from '@xyflow/react'
import ZoomControls from './ZoomControls'
import EmptyState from './EmptyState'

function CanvasArea({
  nodes, edges, nodeTypes, defaultEdgeOptions,
  onNodesChange, onEdgesChange, onConnect,
  onNodeDrag, onNodeDragStop,
  isValidConnection,
  onNodeClick, onPaneClick,
  onDragOver, onDrop,
  snapToGridEnabled,
  isDragging, setIsDragging,
  dragGuides, getViewport,
  canvasIssue,
  hasSelection, selectedNodesCount, selectedEdgesCount,
  handleAlignHorizontal, handleAlignVertical,
  handleTidySelection, handleDeleteSelection,
  isCanvasEmpty, handleLoadExample,
  setNodes, reactFlowWrapper,
}) {
  const handleDragEnter = useCallback(() => setIsDragging(true), [setIsDragging])
  const handleDragLeave = useCallback((e) => {
    if (!e.currentTarget.contains(e.relatedTarget)) setIsDragging(false)
  }, [setIsDragging])

  const viewport = getViewport()
  const zoom = Number(viewport?.zoom) || 1
  const tx = Number(viewport?.x) || 0
  const ty = Number(viewport?.y) || 0

  return (
    <div className="canvas" ref={reactFlowWrapper}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
    >
      <ReactFlow
        nodes={nodes} edges={edges} nodeTypes={nodeTypes}
        defaultEdgeOptions={defaultEdgeOptions}
        onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
        onConnect={onConnect} onNodeDrag={onNodeDrag} onNodeDragStop={onNodeDragStop}
        isValidConnection={isValidConnection}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        onDragOver={onDragOver} onDrop={onDrop}
        fitView snapToGrid={snapToGridEnabled} snapGrid={[15, 15]}
      >
        <MiniMap pannable zoomable
          style={{ background: 'rgba(10, 22, 40, 0.92)', border: '1px solid rgba(90, 138, 181, 0.45)', borderRadius: '8px' }}
          maskColor="rgba(7, 16, 28, 0.55)" nodeColor="#5a8ab5"
        />
        <Background gap={15} color="rgba(255,255,255,0.12)" />
      </ReactFlow>

      {dragGuides.x != null && <div className="alignment-guide alignment-guide-vertical" style={{ left: `${dragGuides.x * zoom + tx}px` }} />}
      {dragGuides.y != null && <div className="alignment-guide alignment-guide-horizontal" style={{ top: `${dragGuides.y * zoom + ty}px` }} />}

      {canvasIssue && <div className={`canvas-issue-banner canvas-issue-${canvasIssue.tone}`}>{canvasIssue.message}</div>}

      {hasSelection && !isDragging && (
        <div className="canvas-selection-hud" aria-live="polite">
          <span className="canvas-selection-count">{selectedNodesCount} node(s), {selectedEdgesCount} edge(s) selected</span>
          <div className="canvas-selection-actions">
            <button type="button" onClick={handleAlignHorizontal} disabled={selectedNodesCount < 2}>Align H</button>
            <button type="button" onClick={handleAlignVertical} disabled={selectedNodesCount < 2}>Align V</button>
            <button type="button" onClick={handleTidySelection} disabled={selectedNodesCount < 1}>Tidy</button>
            <button type="button" className="danger" onClick={handleDeleteSelection}>Delete</button>
          </div>
        </div>
      )}

      {isCanvasEmpty && !isDragging && <EmptyState onLoadTemplate={handleLoadExample} />}
      <ZoomControls nodes={nodes} edges={edges} setNodes={setNodes} />
    </div>
  )
}

export default memo(CanvasArea)
