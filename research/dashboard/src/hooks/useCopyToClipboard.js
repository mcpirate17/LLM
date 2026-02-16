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
      await navigator.clipboard.writeText(text);
      setCopiedValue(text);
      setTimeout(() => setCopiedValue(null), 1200);
    } catch {
      setCopiedValue(null);
    }
  }, []);

  return [copiedValue, copy];
}
