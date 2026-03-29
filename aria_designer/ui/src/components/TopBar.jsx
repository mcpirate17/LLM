import React, { memo, useMemo, useState } from 'react'
import AriaAvatar from './AriaAvatar'

const stepGlyph = (status, busy = false) => {
  if (busy || status === 'running') return '\u21BB'
  if (status === 'pass') return '\u2713'
  if (status === 'fail') return '\u2715'
  return '\u2022'
}

function TopBar({
  runStatus, stepStatus, workflowStage, validateUi,
  saveState, historyUi, snapToGridEnabled,
  handleValidate, handleCompile, handlePreview, handleDeepRun,
  handleSave, handleExportJson, handleExportPython,
  handleImportFile, handleLoadExample, handleReloadComponents, handleClearCanvas,
  handleAlignHorizontal, handleAlignVertical,
  handleDistributeHorizontal, handleDistributeVertical,
  handleTidySelection,
  undoGraph, redoGraph,
  setSnapToGridEnabled, setShowAskAriaModal,
  setShowShortcuts, setShowHelpPanel, setShowImportDialog,
  hardwareView, setHardwareView, heatmapView, setHeatmapView,
  importInputRef, exampleOptions, evalState,
}) {
  const [arrangeOpen, setArrangeOpen] = useState(false)
  const [fileMenuOpen, setFileMenuOpen] = useState(false)
  const [viewMenuOpen, setViewMenuOpen] = useState(false)

  return (
    <header className="topbar">
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <AriaAvatar
          size={50}
          mood={
            runStatus.phase === 'success' ? 'triumphant' :
            runStatus.phase === 'failed' ? 'frustrated' :
            runStatus.phase === 'running' || runStatus.phase === 'compiling' ? 'excited' :
            'curious'
          }
        />
        <div>
          <h1>Aria Designer</h1>
          <p>Graph authoring workspace for user + Aria co-design</p>
        </div>
      </div>
      <div className="actions">
        <input
          ref={importInputRef}
          type="file"
          accept="application/json"
          onChange={handleImportFile}
          style={{ display: 'none' }}
        />
        <div className="toolbar-group workflow">
          <button
            type="button"
            className={`step-btn step-state-${stepStatus.validate} ${workflowStage === 'validate' ? 'active' : ''} ${validateUi.inProgress ? 'busy' : ''}`}
            onClick={handleValidate}
            disabled={validateUi.inProgress}
            title="Step 1: Verify graph structure, ports, and parameters without execution."
          >
            <span className="step-label-row">
              <span className={`step-glyph ${(validateUi.inProgress || stepStatus.validate === 'running') ? 'step-glyph-spin' : ''}`}>
                {stepGlyph(stepStatus.validate, validateUi.inProgress)}
              </span>
              <span>{validateUi.inProgress ? 'Step 1: Validating...' : 'Step 1: Validate'}</span>
            </span>
          </button>
          <span className={`step-sep step-sep-${stepStatus.validate}`} aria-hidden="true">{'\u25B6'}</span>
          <button
            type="button"
            className={`step-btn step-state-${stepStatus.compile} ${workflowStage === 'compile' ? 'active' : ''}`}
            onClick={handleCompile}
            title="Step 2: Convert visual graph into a runnable PyTorch module."
          >
            <span className="step-label-row">
              <span className={`step-glyph ${stepStatus.compile === 'running' ? 'step-glyph-spin' : ''}`}>{stepGlyph(stepStatus.compile)}</span>
              <span>Step 2: Compile</span>
            </span>
          </button>
          <span className={`step-sep step-sep-${stepStatus.compile}`} aria-hidden="true">{'\u25B6'}</span>
          <button
            type="button"
            className={`step-btn step-state-${stepStatus.test} ${workflowStage === 'run' ? 'active' : ''}`}
            onClick={handlePreview}
            title="Step 3: Run forward pass with dummy data to verify shapes and latency."
          >
            <span className="step-label-row">
              <span className={`step-glyph ${stepStatus.test === 'running' ? 'step-glyph-spin' : ''}`}>{stepGlyph(stepStatus.test)}</span>
              <span>Step 3: Test</span>
            </span>
          </button>
          <span className={`step-sep step-sep-${stepStatus.test}`} aria-hidden="true">{'\u25B6'}</span>
          <button
            type="button"
            className={`step-btn step-state-${stepStatus.run} ${(workflowStage === 'deep-run' || evalState.status === 'running') ? 'active' : ''}`}
            onClick={handleDeepRun}
            title="Step 4: Execute full micro-training pipeline (Stage 1) to evaluate loss ratio and novelty."
          >
            <span className="step-label-row">
              <span className={`step-glyph ${stepStatus.run === 'running' ? 'step-glyph-spin' : ''}`}>{stepGlyph(stepStatus.run)}</span>
              <span>Step 4: Run</span>
            </span>
          </button>
        </div>
        <div className="toolbar-group files">
          <div className="arrange-dropdown-wrap">
            <button type="button" onClick={() => setFileMenuOpen((v) => !v)} className={fileMenuOpen ? 'active' : ''} title="File operations">
              File {'\u25BE'}
            </button>
            {fileMenuOpen && (
              <div className="arrange-dropdown" onMouseLeave={() => setFileMenuOpen(false)}>
                <button type="button" onClick={() => { handleSave(); setFileMenuOpen(false) }} disabled={saveState.phase === 'saving'}>
                  {saveState.phase === 'saving' ? 'Saving\u2026' : 'Save'}
                </button>
                <hr />
                <button type="button" onClick={() => { handleExportJson(); setFileMenuOpen(false) }}>Export JSON</button>
                <button type="button" onClick={() => { handleExportPython(); setFileMenuOpen(false) }}>Export Python</button>
                <hr />
                <button type="button" onClick={() => { importInputRef.current?.click(); setFileMenuOpen(false) }}>Import JSON</button>
                <button type="button" onClick={() => { setShowImportDialog(true); setFileMenuOpen(false) }}>Import Research</button>
                <hr />
                {exampleOptions.map((ex) => (
                  <button key={ex.value} type="button" onClick={() => { handleLoadExample(ex.value); setFileMenuOpen(false) }}>
                    Example: {ex.label}
                  </button>
                ))}
                <hr />
                <button type="button" onClick={() => { handleReloadComponents(); setFileMenuOpen(false) }}>Reload Components</button>
                <button type="button" onClick={() => { handleClearCanvas(); setFileMenuOpen(false) }} style={{ color: '#ff5050' }}>Clear Canvas</button>
              </div>
            )}
          </div>
          {saveState.phase !== 'idle' && (
            <span
              className={`save-feedback save-${saveState.phase}`}
              title={saveState.fingerprint ? `${saveState.message}\n${saveState.fingerprint}` : saveState.message}
            >
              {saveState.discoveryUrl && saveState.fingerprint ? (
                <>
                  Saved {'\u00b7'} fp{' '}
                  <a
                    href={saveState.discoveryUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="fingerprint-link"
                  >
                    {String(saveState.fingerprint).slice(0, 12)}...
                  </a>
                </>
              ) : (
                saveState.message
              )}
            </span>
          )}
        </div>
        <div className="toolbar-group library">
          <button type="button" onClick={undoGraph} disabled={!historyUi.canUndo} title="Undo (Ctrl+Z)">Undo</button>
          <button type="button" onClick={redoGraph} disabled={!historyUi.canRedo} title="Redo (Ctrl+Shift+Z)">Redo</button>
          <div className="arrange-dropdown-wrap">
            <button type="button" onClick={() => setArrangeOpen((v) => !v)} className={arrangeOpen ? 'active' : ''} title="Arrange and align nodes">
              Arrange {'\u25BE'}
            </button>
            {arrangeOpen && (
              <div className="arrange-dropdown" onMouseLeave={() => setArrangeOpen(false)}>
                <button type="button" onClick={() => { handleAlignHorizontal(); setArrangeOpen(false) }}>Align Horizontal</button>
                <button type="button" onClick={() => { handleAlignVertical(); setArrangeOpen(false) }}>Align Vertical</button>
                <button type="button" onClick={() => { handleDistributeHorizontal(); setArrangeOpen(false) }}>Distribute Horizontal</button>
                <button type="button" onClick={() => { handleDistributeVertical(); setArrangeOpen(false) }}>Distribute Vertical</button>
                <hr />
                <button type="button" onClick={() => { handleTidySelection(); setArrangeOpen(false) }}>Tidy Selection</button>
                <hr />
                <button type="button" onClick={() => { setSnapToGridEnabled((v) => !v); setArrangeOpen(false) }}>
                  Snap to Grid: {snapToGridEnabled ? 'ON' : 'OFF'}
                </button>
              </div>
            )}
          </div>
        </div>
        {workflowStage !== 'idle' && runStatus.phase !== 'idle' && (
          <div className={`validation-banner ${
            ['running', 'compiling'].includes(runStatus.phase) ? 'running'
            : runStatus.phase === 'success' ? 'pass'
            : runStatus.phase === 'failed' ? 'fail'
            : 'running'
          }`}>
            {runStatus.message}
          </div>
        )}
        <div className="toolbar-group ai">
          <div className="arrange-dropdown-wrap">
            <button type="button" onClick={() => setViewMenuOpen((v) => !v)} className={viewMenuOpen ? 'active' : ''} title="View options">
              View {'\u25BE'}
            </button>
            {viewMenuOpen && (
              <div className="arrange-dropdown" onMouseLeave={() => setViewMenuOpen(false)}>
                <button type="button" onClick={() => { setHardwareView((v) => !v); setViewMenuOpen(false) }}>
                  {hardwareView ? '\u2713 ' : '\u2003 '}Hardware View
                </button>
                <button type="button" onClick={() => { setHeatmapView((v) => !v); setViewMenuOpen(false) }}>
                  {heatmapView ? '\u2713 ' : '\u2003 '}Heatmap
                </button>
                <hr />
                <button type="button" onClick={() => { setShowShortcuts(true); setViewMenuOpen(false) }}>Keyboard Shortcuts</button>
                <button type="button" onClick={() => { setShowHelpPanel(true); setViewMenuOpen(false) }}>Help &amp; Guide</button>
              </div>
            )}
          </div>
          <button type="button" className="primary" onClick={() => setShowAskAriaModal(true)}>Ask Aria</button>
        </div>
      </div>
    </header>
  )
}

export default memo(TopBar)
