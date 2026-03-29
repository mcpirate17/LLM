import { lazy, memo, Suspense } from 'react'

const AskAriaModal = lazy(() => import('./AskAriaModal'))
const NexusCommandPalette = lazy(() => import('./NexusCommandPalette'))
const KeyboardShortcuts = lazy(() => import('./KeyboardShortcuts'))
const HelpPanel = lazy(() => import('./HelpPanel'))
const ImportDialog = lazy(() => import('./ImportDialog'))

const LazyFallback = null

function ModalLayer({
  showAskAriaModal, setShowAskAriaModal,
  handleAskAriaSubmit, handleAskAriaSuggest,
  ariaSuggestions, ariaLoading, setAriaSuggestions,
  setRightPanelTab,
  showNexusPalette, setShowNexusPalette,
  components, handleNexusAction,
  showShortcuts, setShowShortcuts,
  showHelpPanel, setShowHelpPanel,
  showImportDialog, setShowImportDialog,
  loadWorkflowJson,
}) {
  return (
    <Suspense fallback={LazyFallback}>
      {showAskAriaModal && (
        <AskAriaModal
          open={showAskAriaModal}
          onClose={() => { setShowAskAriaModal(false); setAriaSuggestions([]) }}
          onSubmitPrompt={handleAskAriaSubmit}
          onSuggest={handleAskAriaSuggest}
          onSwitchToChat={() => { setShowAskAriaModal(false); setAriaSuggestions([]); setRightPanelTab('chat') }}
          suggestions={ariaSuggestions}
          loading={ariaLoading}
        />
      )}
      {showNexusPalette && (
        <NexusCommandPalette open={showNexusPalette} onClose={() => setShowNexusPalette(false)}
          components={components} onAction={handleNexusAction} />
      )}
      {showShortcuts && <KeyboardShortcuts onClose={() => setShowShortcuts(false)} />}
      {showHelpPanel && <HelpPanel isOpen={showHelpPanel} onClose={() => setShowHelpPanel(false)} />}
      {showImportDialog && (
        <ImportDialog onImport={(wf) => loadWorkflowJson(wf)} onClose={() => setShowImportDialog(false)} />
      )}
    </Suspense>
  )
}

export default memo(ModalLayer)
