import React, { useMemo, useState } from 'react';
import { filterRowsByQuery } from '../../utils/tableFiltering';

export function useSortableFilteredRows(rows, filterFields, initialSortKey = 'count', initialSortDesc = true) {
  const [sortKey, setSortKey] = useState(initialSortKey);
  const [sortDesc, setSortDesc] = useState(initialSortDesc);
  const [filterQuery, setFilterQuery] = useState('');

  const filtered = useMemo(
    () => filterRowsByQuery(rows || [], filterQuery, filterFields || []),
    [rows, filterQuery, filterFields]
  );

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      const va = a?.[sortKey];
      const vb = b?.[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'string') return sortDesc ? vb.localeCompare(va) : va.localeCompare(vb);
      return sortDesc ? vb - va : va - vb;
    });
    return arr;
  }, [filtered, sortKey, sortDesc]);

  const handleSort = (key) => {
    if (sortKey === key) setSortDesc((prev) => !prev);
    else {
      setSortKey(key);
      setSortDesc(true);
    }
  };

  return {
    sortKey,
    sortDesc,
    filterQuery,
    setFilterQuery,
    sorted,
    handleSort,
  };
}

export function ReportFilterRow({ value, onChange, placeholder }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: 8 }}>
      <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Filter:</div>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        style={{
          fontSize: 11,
          padding: '4px 8px',
          borderRadius: 4,
          border: '1px solid var(--border)',
          background: 'var(--bg-tertiary)',
          color: 'var(--text-primary)',
          minWidth: 160,
        }}
      />
    </div>
  );
}

export function SortableHeader({ label, sortKey, activeSortKey, sortDesc, onSort, style }) {
  return (
    <th onClick={() => onSort(sortKey)} style={style}>
      {label}
      {activeSortKey === sortKey && (
        <span style={{ marginLeft: 4, fontSize: 10 }}>{sortDesc ? '\u25BC' : '\u25B2'}</span>
      )}
    </th>
  );
}
