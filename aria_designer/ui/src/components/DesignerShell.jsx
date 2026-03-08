import React, { useState, useCallback, useEffect, useRef } from 'react'
import AriaAvatar from './AriaAvatar'
import AskAriaModal from './AskAriaModal'
import NexusCommandPalette from './NexusCommandPalette'
import KeyboardShortcuts from './KeyboardShortcuts'
import ImportDialog from './ImportDialog'
import RunResultsPanel from './RunResultsPanel'
import PatchPanel from './PatchPanel'
import Inspector from './Inspector'
import Palette from './Palette'
import ErrorBoundary from './ErrorBoundary'

export function DesignerShell({
  children,
  state,
  actions,
  components,
  paletteConstraints,
  workflowStage,
  stepStatus,
  runStatus,
  evalState,
  saveState,
  importedBaseline,
  benchmarkObserved,
  setBenchmarkObserved,
  selectedNode,
  onParamChange,
  helpRequest,
  historyUi,
  undoGraph,
  redoGraph,
  handleAlignHorizontal,
  handleAlignVertical,
  handleDistributeHorizontal,
  handleDistributeVertical,
  handleTidySelection,
  handleAskAriaSubmit,
  handleAskAriaSuggest,
  ariaSuggestions,
  ariaLoading,
  handleNexusAction,
  loadWorkflowJson,
  handleExportJson,
  handleExportPython,
  handleReloadComponents,
  handleClearCanvas,
  handleImportFile,
  importInputRef,
  exampleOptions,
  handleLoadExample,
  handleApplyPatch,
  handleRejectPatch,
  handlePreviewPatch,
  scopedProposals,
  embeddedMode,
  readOnly
}) {
  const {
    rightPanelWidth, setRightPanelWidth,
    rightPanelTab, setRightPanelTab,
    statusMsg, toasts, nodes, edges,
    snapToGridEnabled, setSnapToGridEnabled,
    hardwareView, setHardwareView,
    heatmapView, setHeatmapView,
    setShowShortcuts, showShortcuts
  } = state

  const [isResizing, setIsResizing] = useState(false)
  const [fileMenuOpen, setFileMenuOpen] = useState(false)
  const [arrangeOpen, setArrangeOpen] = useState(false)
  const [viewMenuOpen, setViewMenuOpen] = useState(false)
  const [showAskAriaModal, setShowAskAriaModal] = useState(false)
  const [showNexusPalette, setShowNexusPalette] = useState(false)
  const [showImportDialog, setShowImportDialog] = useState(false)

  const resizeRef = useRef({ startX: 0, startWidth: 300 })

  const startResizing = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    resizeRef.current = { startX: e.clientX, startWidth: rightPanelWidth }
    setIsResizing(true)
  }, [rightPanelWidth])

  useEffect(() => {
    if (!isResizing) return
    const handleMouseMove = (e) => {
      const deltaX = resizeRef.current.startX - e.clientX
      setRightPanelWidth(Math.max(250, Math.min(900, resizeRef.current.startWidth + deltaX)))
    }
    const handleMouseUp = () => setIsResizing(false)
    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isResizing, setRightPanelWidth])

  const handleResizeKeyDown = useCallback((e) => {
    const step = e.shiftKey ? 40 : 16
    if (e.key === 'ArrowLeft') setRightPanelWidth((w) => Math.max(250, Math.min(900, w + step)))
    if (e.key === 'ArrowRight') setRightPanelWidth((w) => Math.max(250, Math.min(900, w - step)))
  }, [setRightPanelWidth])

  const stepGlyph = (status, busy = false) => {
    if (busy || status === 'running') return '↻'
    if (status === 'pass') return '✓'
    if (status === 'fail') return '✕'
    return '•'
  }

  return (
    <div className={`page ${embeddedMode ? 'embedded-mode' : ''} ${isResizing ? 'is-resizing' : ''}`} style={{ '--right-panel-width': `${rightPanelWidth}px` }}>
      {!embeddedMode && (
        <Palette
          components={components}
          onDragStart={() => {}}
          constraints={paletteConstraints}
        />
      )}

      <main className="canvas-wrap">
        {!embeddedMode && (
          <header className="topbar">
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              <AriaAvatar size={50} mood={runStatus.phase === 'success' ? 'triumphant' : 'curious'} />
              <div>
                <h1>Aria Designer</h1>
                <p>Graph authoring workspace</p>
              </div>
            </div>
            <div className="actions">
              <input ref={importInputRef} type="file" accept="application/json" onChange={handleImportFile} style={{ display: 'none' }} />
              <div className="toolbar-group workflow">
                <button className={`step-btn step-state-${stepStatus.validate}`} onClick={actions.handleValidate}>
                  {stepGlyph(stepStatus.validate)} Validate
                </button>
                <button className={`step-btn step-state-${stepStatus.compile}`} onClick={actions.handleSave}>
                  {stepGlyph(stepStatus.compile)} Save
                </button>
              </div>
              <div className="toolbar-group files">
                <button onClick={() => setFileMenuOpen(!fileMenuOpen)}>File ▾</button>
                {fileMenuOpen && (
                  <div className="arrange-dropdown" onMouseLeave={() => setFileMenuOpen(false)}>
                    <button onClick={actions.handleSave}>Save</button>
                    <button onClick={handleExportJson}>Export JSON</button>
                    <button onClick={() => setShowImportDialog(true)}>Import Research</button>
                    <button onClick={handleClearCanvas} style={{ color: 'red' }}>Clear Canvas</button>
                  </div>
                )}
              </div>
              <div className="toolbar-group library">
                <button onClick={undoGraph} disabled={!historyUi.canUndo}>Undo</button>
                <button onClick={redoGraph} disabled={!historyUi.canRedo}>Redo</button>
              </div>
              <button className="primary" onClick={() => setShowAskAriaModal(true)}>Ask Aria</button>
            </div>
          </header>
        )}

        <div className="canvas-container-inner" style={{ position: 'relative', flex: 1 }}>
          {children}
        </div>

        {!embeddedMode && (
          <footer className="statusbar">
            <div className="status-block">
              <span className={`run-chip run-${runStatus.phase}`}>{runStatus.phase.toUpperCase()}</span>
              <span className="status-msg">{runStatus.message}</span>
            </div>
            <div className="status-block secondary">
              <span>{nodes.length} nodes</span>
              <span>{edges.length} edges</span>
            </div>
          </footer>
        )}

        <div className="toast-stack">
          {toasts.map((t) => (
            <div key={t.id} className={`toast toast-${t.tone || 'info'}`}>{t.message}</div>
          ))}
        </div>
      </main>

      {!embeddedMode && (
        <aside className="panel right">
          <div className="resize-handle-left" onMouseDown={startResizing} onKeyDown={handleResizeKeyDown} tabIndex={0} />
          <div className="panel-tabs">
            <button className={rightPanelTab === 'inspector' ? 'active' : ''} onClick={() => setRightPanelTab('inspector')}>Properties</button>
            <button className={rightPanelTab === 'results' ? 'active' : ''} onClick={() => setRightPanelTab('results')}>Results</button>
          </div>
          <div className="panel-content">
            {rightPanelTab === 'results' ? (
              <RunResultsPanel evalState={evalState} baseline={importedBaseline} benchmarkObserved={benchmarkObserved} onBenchmarkObservedChange={setBenchmarkObserved} />
            ) : (
              <Inspector selectedNode={selectedNode} allComponents={components} onParamChange={onParamChange} helpRequest={helpRequest} />
            )}
          </div>
        </aside>
      )}

      <AskAriaModal open={showAskAriaModal} onClose={() => setShowAskAriaModal(false)} onSubmitPrompt={handleAskAriaSubmit} onSuggest={handleAskAriaSuggest} suggestions={ariaSuggestions} loading={ariaLoading} />
      <NexusCommandPalette open={showNexusPalette} onClose={() => setShowNexusPalette(false)} components={components} onAction={handleNexusAction} />
      {showShortcuts && <KeyboardShortcuts onClose={() => setShowShortcuts(false)} />}
      {showImportDialog && <ImportDialog onImport={loadWorkflowJson} onClose={() => setShowImportDialog(false)} />}
    </div>
  )
}
