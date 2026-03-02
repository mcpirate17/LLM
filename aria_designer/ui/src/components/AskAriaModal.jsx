import { memo, useEffect, useState } from 'react'

const INTENT_PRESETS = [
  {
    id: 'split_pipeline',
    label: 'Split Pipeline',
    prompt: 'Split this pipeline into two parallel branches and merge them back safely near the output.',
  },
  {
    id: 'add_routing',
    label: 'Add Routing',
    prompt: 'Add a routing mechanism (top-k or early-exit style) to improve efficiency while preserving stability.',
  },
  {
    id: 'add_compression',
    label: 'Add Compression',
    prompt: 'Add a compression block (low-rank/bottleneck style) to reduce parameter and FLOP cost with minimal quality loss.',
  },
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
    id: 'beat_benchmarks',
    label: 'Beat Benchmarks',
    prompt: 'Propose a patch that closes benchmark-target gaps for speed, FLOPs, novelty, and downstream task quality while preserving stability.',
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
  {
    id: 'optimize_data_control',
    label: 'Optimize Data/Control',
    prompt: 'Suggest data/control workflow optimizations focused on join/filter behavior and schema hygiene (column selection, deterministic filtering, explicit control-flow guards).',
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
  const sortedSuggestions = [...(suggestions || [])]
    .sort((a, b) => (b?.score || 0) - (a?.score || 0))
  const hasPrompt = Boolean(prompt.trim())

  useEffect(() => {
    if (!open) {
      setSelectedPreset('')
      setPrompt('')
    }
  }, [open])

  useEffect(() => {
    if (!open) return
    const onKeyDown = (e) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [open, onClose])

  if (!open) return null

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content ask-aria-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Ask Aria</h2>
          <button type="button" className="close-btn" onClick={onClose}>&times;</button>
        </div>

        <p className="muted">Describe the change you want. Aria will rank suggestions from your current graph and draft a deterministic patch.</p>

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
                aria-pressed={selectedPreset === p.id}
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
          placeholder="Example: Improve stability and reduce FLOPs with minimal quality loss."
          rows={5}
        />
        <div className="ask-prompt-meta">
          <span>{prompt.trim().length} chars</span>
        </div>

        <div className="ask-actions">
          <button type="button" onClick={() => onSuggest(prompt)} disabled={loading || !hasPrompt}>
            {loading ? 'Scoring graph…' : 'Get Data-Driven Suggestions'}
          </button>
          <button type="button" className="primary" onClick={() => onSubmitPrompt(prompt)} disabled={loading || !hasPrompt}>
            Create Proposal
          </button>
        </div>

        <div className="ask-suggestions">
          <div className="ask-suggestions-header">
            <h3>Suggestions</h3>
            <span className="ask-suggestions-meta">
              {sortedSuggestions.length > 0 ? `${sortedSuggestions.length} ranked` : 'awaiting score pass'}
            </span>
          </div>
          {loading ? (
            <div className="loading-placeholder-list" aria-hidden="true">
              {[1, 2, 3, 4].map((row) => (
                <div key={row} className="loading-placeholder-card">
                  <div className="loading-placeholder-line short" />
                  <div className="loading-placeholder-line mid" />
                  <div className="loading-placeholder-line long" />
                </div>
              ))}
            </div>
          ) : sortedSuggestions.length === 0 ? (
            <p className="muted">No suggestions yet. Enter a prompt and run a scoring pass.</p>
          ) : (
            <div className="ask-suggestion-list">
              {sortedSuggestions.map((s, idx) => {
                const compName = s.component?.name || s.component?.id || 'Unknown'
                const compType = s.component?.category ? `${s.component.category}/${s.component?.id}` : (s.component?.id || '')
                const scorePct = Math.round((s.score || 0) * 100)
                return (
                  <div key={`${s.component?.id || 's'}-${idx}`} className="ask-suggestion-card">
                    <div className="ask-suggestion-head">
                      <div className="ask-suggestion-title">{compName}</div>
                      <span className="ask-suggestion-score">{scorePct}%</span>
                    </div>
                    <div className="ask-suggestion-type">{compType}</div>
                    <div className="ask-suggestion-reason">{s.reason || 'Suggested by Aria'}</div>
                    {Array.isArray(s.evidence) && s.evidence.length > 0 && (
                      <ul className="ask-suggestion-evidence">
                        {s.evidence.slice(0, 2).map((ev, i) => (
                          <li key={`${compType}-${i}`}>{ev}</li>
                        ))}
                      </ul>
                    )}
                    <div className="ask-suggestion-actions">
                      <button
                        type="button"
                        onClick={() => setPrompt(`Add ${compType} and connect it at the best leaf node. Reason: ${s.reason || 'improve graph quality'}.`)}
                      >
                        Use
                      </button>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default memo(AskAriaModal)
