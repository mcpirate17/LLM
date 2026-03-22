import { useState, useEffect } from 'react';

/**
 * useState backed by localStorage. Reads initial value from storage,
 * writes back on every state change.
 *
 * @param {string} key - localStorage key
 * @param {*} defaultValue - fallback when key is missing or unparseable
 * @returns {[*, Function]} - [value, setValue] identical to useState
 */
export default function useLocalStorage(key, defaultValue) {
  const [value, setValue] = useState(() => {
    try {
      const stored = window.localStorage.getItem(key);
      if (stored === null) return defaultValue;
      return JSON.parse(stored);
    } catch {
      return defaultValue;
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // storage full or unavailable — ignore
    }
  }, [key, value]);

  return [value, setValue];
}
