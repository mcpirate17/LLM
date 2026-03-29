import { lazy, memo, Suspense, useCallback } from 'react'
import ErrorBoundary from './ErrorBoundary'
import Inspector from './Inspector'
import PatchPanel from './PatchPanel'

const AriaChatPanel = lazy(() => import('./AriaChatPanel'))
const RunResultsPanel = lazy(() => import('./RunResultsPanel'))

const LazyFallback = <div className="lazy-loading">Loading...</div>

function RightPanel({
  rightPanelTab, setRightPanelTab,
  rightPanelWidth, isResizing, startResizing, handleResizeKeyDown,
  // Inspector props
  selectedNode, components, nodes, edges, onParamChange, helpRequest,
  // Results props
  evalState, importedBaseline, benchmarkObserved, setBenchmarkObserved,
  // Chat props
  getWorkflowJsonForChat, handleChatApplyPatch,
  // Proposals props
  scopedProposals, handleApplyPatch, handleRejectPatch, handlePreviewPatch,
  setPreviewPatch, setNodes,
}) {
  const switchToInspector = useCallback(() => {
    setRightPanelTab('inspector')
    setPreviewPatch(null)
    setNodes(nds => nds.map(n => ({ ...n, className: '' })))
  }, [setRightPanelTab, setPreviewPatch, setNodes])

  return (
    <aside className="panel right">
      <div
        className={`resize-handle-left ${isResizing ? 'resizing' : ''}`}
        onMouseDown={startResizing} onKeyDown={handleResizeKeyDown}
        role="separator" aria-orientation="vertical" aria-label="Resize properties panel"
        aria-valuemin={250} aria-valuemax={900} aria-valuenow={Math.round(rightPanelWidth)} tabIndex={0}
        title="Drag to resize properties panel"
      />
      <div className="panel-tabs">
        <button type="button" className={rightPanelTab === 'inspector' ? 'active' : ''} aria-pressed={rightPanelTab === 'inspector'}
          onClick={switchToInspector}>
          Properties
        </button>
        <button type="button" className={rightPanelTab === 'chat' ? 'active' : ''} aria-pressed={rightPanelTab === 'chat'}
          onClick={() => setRightPanelTab('chat')}>
          Aria Chat
        </button>
        {scopedProposals.length > 0 && (
          <button type="button" className={rightPanelTab === 'proposals' ? 'active' : ''} aria-pressed={rightPanelTab === 'proposals'}
            onClick={() => setRightPanelTab('proposals')}>
            Proposals ({scopedProposals.length})
          </button>
        )}
        <button type="button" className={rightPanelTab === 'results' ? 'active' : ''} aria-pressed={rightPanelTab === 'results'}
          onClick={() => setRightPanelTab('results')}>
          Results
        </button>
      </div>

      {rightPanelTab === 'results' ? (
        <Suspense fallback={LazyFallback}>
          <RunResultsPanel evalState={evalState} baseline={importedBaseline}
            benchmarkObserved={benchmarkObserved} onBenchmarkObservedChange={setBenchmarkObserved} />
        </Suspense>
      ) : rightPanelTab === 'chat' ? (
        <ErrorBoundary name="Chat">
          <Suspense fallback={LazyFallback}>
            <AriaChatPanel
              workflowJsonFn={getWorkflowJsonForChat}
              onApplyPatch={handleChatApplyPatch}
            />
          </Suspense>
        </ErrorBoundary>
      ) : rightPanelTab === 'proposals' ? (
        <PatchPanel proposals={scopedProposals} onApply={handleApplyPatch} onReject={handleRejectPatch}
          onPreview={handlePreviewPatch}
          onClose={switchToInspector} />
      ) : (
        <ErrorBoundary name="Inspector">
          <Inspector selectedNode={selectedNode} allComponents={components} nodeCount={nodes.length}
            edgeCount={edges.length} onParamChange={onParamChange} helpRequest={helpRequest} />
        </ErrorBoundary>
      )}
    </aside>
  )
}

export default memo(RightPanel)
