import { useEffect } from 'react';

const TAB_KEYS = ['command', 'trends', 'experiments', 'discoveries', 'perf', 'reports', 'log'];

export default function useKeyboardShortcuts({
  showHelp,
  setShowHelp,
  showChat,
  setShowChat,
  showSettings,
  setShowSettings,
  selectedProgram,
  closeSelectedProgram,
  designerSession,
  closeDesigner,
  setActiveTab,
  setSelectedExperiment,
}) {
  useEffect(() => {
    const handler = (e) => {
      const tag = (e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select' || e.target.isContentEditable) return;

      if (e.key === '?') { e.preventDefault(); setShowHelp(h => !h); return; }
      if (e.key === 'Escape') {
        if (showHelp) { setShowHelp(false); return; }
        if (showChat) { setShowChat(false); return; }
        if (showSettings) { setShowSettings(false); return; }
        if (designerSession.open) { closeDesigner(); return; }
        if (selectedProgram) { closeSelectedProgram(); return; }
        return;
      }
      const num = parseInt(e.key, 10);
      if (num >= 1 && num <= TAB_KEYS.length) {
        e.preventDefault();
        setActiveTab(TAB_KEYS[num - 1]);
        setSelectedExperiment(null);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [
    showHelp, showChat, showSettings, selectedProgram,
    designerSession.open, closeDesigner, closeSelectedProgram,
    setShowHelp, setShowChat, setShowSettings, setActiveTab, setSelectedExperiment,
  ]);
}
