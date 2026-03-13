import { useCallback, useEffect, useRef, useState } from 'react';

export default function useResizableColumns(storageKey) {
  const [columnWidths, setColumnWidths] = useState(() => {
    try {
      const saved = window.localStorage.getItem(storageKey);
      return saved ? JSON.parse(saved) : {};
    } catch {
      return {};
    }
  });
  const resizingRef = useRef(null);

  useEffect(() => {
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(columnWidths));
    } catch {
      // ignore localStorage failures in private browsing or SSR
    }
  }, [storageKey, columnWidths]);

  const onResizeStart = useCallback((e, colKey) => {
    e.preventDefault();
    e.stopPropagation();
    const startX = e.clientX;
    const th = e.target.parentElement;
    const startWidth = th.offsetWidth;
    resizingRef.current = colKey;

    const onMouseMove = (moveE) => {
      const diff = moveE.clientX - startX;
      const newWidth = Math.max(40, startWidth + diff);
      setColumnWidths((prev) => ({ ...prev, [colKey]: newWidth }));
    };
    const onMouseUp = () => {
      resizingRef.current = null;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, []);

  return { columnWidths, onResizeStart };
}
