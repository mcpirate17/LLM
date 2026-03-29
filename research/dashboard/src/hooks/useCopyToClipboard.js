import { useState, useCallback } from 'react';

/**
 * Hook for copy-to-clipboard with visual feedback.
 * Returns [copiedValue, copyFn] where copiedValue is the text
 * that was most recently copied (resets after 1200ms), and copyFn
 * copies a string to clipboard.
 */
export default function useCopyToClipboard() {
  const [copiedValue, setCopiedValue] = useState(null);

  const copy = useCallback(async (text) => {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        // Fallback for non-secure contexts (e.g. HTTP on non-localhost)
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
      }
      setCopiedValue(text);
      setTimeout(() => setCopiedValue(null), 1200);
    } catch {
      setCopiedValue(null);
    }
  }, []);

  return [copiedValue, copy];
}
