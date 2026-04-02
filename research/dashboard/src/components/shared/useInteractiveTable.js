import { useEffect, useMemo, useState } from 'react';
import { filterRowsByQuery } from '../../utils/tableFiltering';

export default function useInteractiveTable({
  rows,
  filterFields,
  initialSortKey,
  initialSortDesc = true,
  storageKey,
  getSortValue,
  getInitialSortDesc,
}) {
  const [sortKey, setSortKey] = useState(() => {
    if (!storageKey || typeof window === 'undefined') return initialSortKey;
    try {
      const stored = JSON.parse(window.localStorage.getItem(storageKey) || '{}');
      return typeof stored.sortKey === 'string' ? stored.sortKey : initialSortKey;
    } catch {
      return initialSortKey;
    }
  });
  const [sortDesc, setSortDesc] = useState(() => {
    if (!storageKey || typeof window === 'undefined') return initialSortDesc;
    try {
      const stored = JSON.parse(window.localStorage.getItem(storageKey) || '{}');
      return typeof stored.sortDesc === 'boolean' ? stored.sortDesc : initialSortDesc;
    } catch {
      return initialSortDesc;
    }
  });
  const [filterQuery, setFilterQuery] = useState('');

  useEffect(() => {
    if (!storageKey || typeof window === 'undefined') return;
    try {
      window.localStorage.setItem(storageKey, JSON.stringify({ sortKey, sortDesc }));
    } catch {
      // Ignore localStorage failures.
    }
  }, [sortDesc, sortKey, storageKey]);

  const filteredRows = useMemo(
    () => filterRowsByQuery(rows || [], filterQuery, filterFields || []),
    [filterFields, filterQuery, rows],
  );

  const sortedRows = useMemo(() => {
    const list = [...filteredRows];
    list.sort((a, b) => {
      const left = getSortValue ? getSortValue(a, sortKey) : a?.[sortKey];
      const right = getSortValue ? getSortValue(b, sortKey) : b?.[sortKey];
      if (left == null && right == null) return 0;
      if (left == null) return 1;
      if (right == null) return -1;
      if (typeof left === 'string' || typeof right === 'string') {
        return sortDesc ? String(right).localeCompare(String(left)) : String(left).localeCompare(String(right));
      }
      return sortDesc ? Number(right) - Number(left) : Number(left) - Number(right);
    });
    return list;
  }, [filteredRows, getSortValue, sortDesc, sortKey]);

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDesc((prev) => !prev);
      return;
    }
    setSortKey(key);
    setSortDesc(getInitialSortDesc ? getInitialSortDesc(key) : true);
  };

  return {
    sortKey,
    sortDesc,
    filterQuery,
    setFilterQuery,
    sortedRows,
    handleSort,
  };
}
