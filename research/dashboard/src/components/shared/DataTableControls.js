import React from 'react';

export function TableFilterInput({
  value,
  onChange,
  placeholder,
  minWidth = 160,
  ariaLabel,
}) {
  return (
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      aria-label={ariaLabel || placeholder}
      style={{
        fontSize: 11,
        padding: '4px 8px',
        borderRadius: 4,
        border: '1px solid var(--border)',
        background: 'var(--bg-tertiary)',
        color: 'var(--text-primary)',
        minWidth,
      }}
    />
  );
}

export function SortableHeader({
  sortKey,
  activeSortKey,
  sortDesc,
  onSort,
  label,
  title,
  ariaLabel,
  style,
}) {
  const isActive = activeSortKey === sortKey;

  return (
    <th
      onClick={() => onSort(sortKey)}
      title={title}
      aria-sort={isActive ? (sortDesc ? 'descending' : 'ascending') : 'none'}
      aria-label={ariaLabel || `Sort by ${label}`}
      className="th-sortable"
      style={{ whiteSpace: 'nowrap', ...style }}
    >
      {label}
      {isActive && (
        <span className="th-sort-icon">
          {sortDesc ? '\u25BC' : '\u25B2'}
        </span>
      )}
    </th>
  );
}
