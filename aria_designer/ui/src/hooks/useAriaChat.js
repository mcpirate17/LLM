import { useCallback, useState } from 'react'
import { apiCall } from '../services/apiService'

export function useAriaChat(workflowJsonFn) {
  const [sessionId, setSessionId] = useState(null)
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)

  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || loading) return
    const userMsg = { role: 'user', content: text }
    setMessages((prev) => [...prev, userMsg])
    setLoading(true)
    try {
      const workflow = workflowJsonFn ? workflowJsonFn() : undefined
      const body = {
        message: text,
        session_id: sessionId || undefined,
        workflow: workflow || undefined,
      }
      const res = await apiCall('/api/v1/aria/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await res.json()
      if (data.session_id) setSessionId(data.session_id)
      setMessages((prev) => [...prev, {
        role: 'aria',
        content: data.content,
        suggestions: data.suggestions,
        patchProposal: data.patch_proposal,
        needsClarification: data.needs_clarification,
      }])
    } catch {
      setMessages((prev) => [...prev, { role: 'aria', content: 'Sorry, I encountered an error. Please try again.' }])
    } finally {
      setLoading(false)
    }
  }, [sessionId, loading, workflowJsonFn])

  const resetChat = useCallback(() => {
    if (sessionId) {
      apiCall(`/api/v1/aria/chat/${sessionId}`, { method: 'DELETE' }).catch(() => {})
    }
    setSessionId(null)
    setMessages([])
  }, [sessionId])

  return { messages, sendMessage, loading, resetChat, sessionId }
}
