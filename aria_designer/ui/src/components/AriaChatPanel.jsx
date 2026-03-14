import { memo, useCallback, useRef, useEffect, useState } from 'react'
import { useAriaChat } from '../hooks/useAriaChat'
import '../styles/Chat.css'

function AriaChatPanel({ workflowJsonFn, onApplyPatch }) {
  const { messages, sendMessage, loading, resetChat } = useAriaChat(workflowJsonFn)
  const [input, setInput] = useState('')
  const scrollRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  // Auto-resize textarea to fit content
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 120) + 'px'
  }, [input])

  const handleSend = useCallback(() => {
    if (!input.trim()) return
    sendMessage(input)
    setInput('')
  }, [input, sendMessage])

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }, [handleSend])

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <span className="chat-header-title">Chat with Aria</span>
        {messages.length > 0 && (
          <button type="button" onClick={resetChat} className="chat-header-clear">
            Clear
          </button>
        )}
      </div>
      <div className="chat-messages" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="chat-empty">
            Describe your architecture goals and Aria will help you build iteratively.
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div className={`chat-bubble ${m.role === 'user' ? 'user' : 'aria'}`}>
              <div className="chat-role">
                {m.role === 'user' ? 'You' : 'Aria'}
              </div>
              {m.content}
            </div>
            {m.patchProposal && (
              <div className="chat-patch-card">
                <div style={{ fontWeight: 600, marginBottom: 6 }}>Patch Proposal</div>
                <div style={{ color: 'var(--muted, #888)', marginBottom: 6 }}>{m.patchProposal.rationale}</div>
                <div style={{ fontSize: 12, color: 'var(--muted, #888)' }}>
                  {m.patchProposal.ops?.length || 0} operation(s)
                </div>
                <div className="chat-patch-card-actions">
                  <button
                    type="button"
                    className="primary"
                    style={{ fontSize: 13, padding: '6px 14px' }}
                    onClick={() => onApplyPatch?.(m.patchProposal)}
                  >
                    Apply Patch
                  </button>
                </div>
              </div>
            )}
            {m.suggestions?.length > 0 && (
              <div className="chat-patch-card">
                <div style={{ fontWeight: 600, marginBottom: 6 }}>Suggestions</div>
                {m.suggestions.slice(0, 3).map((s, j) => (
                  <div key={j} style={{ color: 'var(--muted, #888)', marginBottom: 4 }}>
                    {s.component?.name || s.component?.id || `Option ${j + 1}`}
                    {s.reason ? ` — ${s.reason}` : ''}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="chat-bubble aria" style={{ opacity: 0.6 }}>
            Thinking...
          </div>
        )}
      </div>
      <div className="chat-input-row">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Describe what you want to build..."
          disabled={loading}
          rows={1}
        />
        <button
          type="button"
          className="primary chat-send-btn"
          onClick={handleSend}
          disabled={loading || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  )
}

export default memo(AriaChatPanel)
