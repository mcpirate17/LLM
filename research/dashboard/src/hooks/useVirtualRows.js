import { useState, useCallback, useMemo, useRef } from 'react';

/**
 * Lightweight virtual scroll — renders only visible rows plus overscan buffer.
 * Returns { containerProps, visibleRows, topPadding, bottomPadding, startIndex }.
 * Wrap the scrollable container with {...containerProps} and add padding spacers
 * in the tbody (empty <tr> elements with the appropriate height).
 */
export default function useVirtualRows({ rows, rowHeight = 40, overscan = 10, containerHeight = 600 }) {
  const [scrollTop, setScrollTop] = useState(0);
  const containerRef = useRef(null);

  const onScroll = useCallback((e) => {
    setScrollTop(e.currentTarget.scrollTop);
  }, []);

  const { visibleRows, startIndex, endIndex } = useMemo(() => {
    const totalRows = rows.length;
    const start = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
    const visibleCount = Math.ceil(containerHeight / rowHeight) + 2 * overscan;
    const end = Math.min(totalRows, start + visibleCount);
    return {
      visibleRows: rows.slice(start, end),
      startIndex: start,
      endIndex: end,
    };
  }, [rows, scrollTop, rowHeight, overscan, containerHeight]);

  const topPadding = startIndex * rowHeight;
  const bottomPadding = Math.max(0, (rows.length - endIndex) * rowHeight);

  const containerProps = {
    ref: containerRef,
    onScroll,
    style: { maxHeight: containerHeight, overflowY: 'auto' },
  };

  return { containerProps, visibleRows, topPadding, bottomPadding, startIndex };
}
