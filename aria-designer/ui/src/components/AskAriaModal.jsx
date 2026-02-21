import { memo, useEffect, useState } from 'react'

const INTENT_PRESETS = [
  {
    id: 'refine_fingerprint',
    label: 'Refine Fingerprint',
    prompt: 'Refine this architecture fingerprint while preserving overall structure. Propose small, high-signal operator improvements.',
  },
  {
    id: 'refine_recommended',
    label: 'Refine Recommended',
    prompt: 'Use diagnostics/recommendations from current graph health and propose the highest-priority refinement patch.',
  },
  {
    id: 'refine_compression',
    label: 'Refine Compression',
    prompt: 'Propose a patch that improves compression/efficiency (params/FLOPs/memory) with minimal quality regression risk.',
  },
  {
    id: 'refine_sparsity',
    label: 'Refine Sparsity',
    prompt: 'Propose sparsity-oriented changes (structured sparsity-friendly ops) while preserving gradient flow and stability.',
  },
  {
    id: 'investigate',
    label: 'Investigate',
    prompt: 'Propose an investigation-oriented variant to stress-test a key architectural hypothesis and improve novelty evidence.',
  },
]

function AskAriaModal({
  open,
  onClose,
  onSubmitPrompt,
  onSuggest,
  suggestions = [],
  loading = false,
}) {
  const [prompt, setPrompt] = useState('')
  const [selectedPreset, setSelectedPreset] = useState('')

  useEffect(() => {
    if (!open) {
      setSelectedPreset('')
      setPrompt('')
    }
  }, [open])

  if (!open) return null

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content ask-aria-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Ask Aria</h2>
          <button className="close-btn" onClick={onClose}>&times;</button>
        </div>

        <p className="muted">Describe the graph change you want. Aria will draft a proposal.</p>

        <div className="ask-intent-row">
          <label htmlFor="ask-aria-intent">Quick intent</label>
          <select
            id="ask-aria-intent"
            value={selectedPreset}
            onChange={(e) => {
              const value = e.target.value
              setSelectedPreset(value)
              const preset = INTENT_PRESETS.find((p) => p.id === value)
              if (preset) setPrompt(preset.prompt)
            }}
          >
            <option value="">Custom prompt…</option>
            {INTENT_PRESETS.map((p) => (
              <option key={p.id} value={p.id}>{p.label}</option>
            ))}
          </select>
          <div className="ask-intent-chips">
            {INTENT_PRESETS.map((p) => (
              <button
                key={p.id}
                type="button"
                className={`ask-chip ${selectedPreset === p.id ? 'active' : ''}`}
                onClick={() => {
                  setSelectedPreset(p.id)
                  setPrompt(p.prompt)
                }}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        <textarea
          className="aria-prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Example: Add an output head and connect it to the last node."
          rows={5}
        />

        <div className="ask-actions">
          <button onClick={() => onSuggest()} disabled={loading}>Get Suggestions</button>
          <button className="primary" onClick={() => onSubmitPrompt(prompt)} disabled={loading || !prompt.trim()}>
            Create Proposal
          </button>
        </div>

        <div className="ask-suggestions">
          <h3>Suggestions</h3>
          {suggestions.length === 0 ? (
            <p className="muted">No suggestions yet.</p>
          ) : (
            <ul>
              {suggestions.map((s, idx) => (
                <li key={`${s.component?.id || 's'}-${idx}`}>
                  <strong>{s.component?.name || s.component?.id || 'Unknown'}</strong>
                  <span> — {s.reason || 'Suggested by Aria'}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  )
}

export default memo(AskAriaModal)
