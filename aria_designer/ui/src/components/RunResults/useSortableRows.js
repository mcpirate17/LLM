import { useMemo, useState } from 'react';

export default function useSortableRows(rows, initialSortCol, initialSortAsc = false) {
  const [sortCol, setSortCol] = useState(initialSortCol);
  const [sortAsc, setSortAsc] = useState(initialSortAsc);

  const sortedRows = useMemo(() => {
    const list = Array.isArray(rows) ? [...rows] : [];
    list.sort((a, b) => {
      const av = a?.[sortCol];
      const bv = b?.[sortCol];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === 'string' || typeof bv === 'string') {
        const left = String(av);
        const right = String(bv);
        return sortAsc ? left.localeCompare(right) : right.localeCompare(left);
      }
      return sortAsc ? Number(av) - Number(bv) : Number(bv) - Number(av);
    });
    return list;
  }, [rows, sortAsc, sortCol]);

  const handleSort = (col) => {
    if (sortCol === col) {
      setSortAsc((prev) => !prev);
      return;
    }
    setSortCol(col);
    setSortAsc(false);
  };

  const sortDirection = (col) => (sortCol === col ? (sortAsc ? 'ascending' : 'descending') : 'none');
  const sortGlyph = (col) => (sortCol === col ? (sortAsc ? '▲' : '▼') : '');

  return {
    rows: sortedRows,
    sortCol,
    sortAsc,
    handleSort,
    sortDirection,
    sortGlyph,
  };
}
