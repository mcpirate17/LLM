import React from 'react';
import SortIndicator from './SortIndicator';
import { hasExactColumnKeys, normalizeColumnKeys } from './columnUtils';

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
      <SortIndicator active={isActive} desc={sortDesc} />
    </th>
  );
}

export function ColumnPickerPanel({
  columns,
  selectedKeys,
  onChange,
  onReset,
  presets = [],
  onPreset,
  requiredKeys = [],
  label = 'Choose visible columns',
}) {
  const selected = new Set(normalizeColumnKeys(columns, selectedKeys, requiredKeys));
  const required = new Set(normalizeColumnKeys(columns, requiredKeys));

  return (
    <div className="column-picker-panel" role="group" aria-label={label}>
      {presets.length > 0 && (
        <div className="column-picker-presets" aria-label="Column presets">
          {presets.map((preset) => {
            const keys = normalizeColumnKeys(columns, preset.columns || [], requiredKeys);
            return (
              <button
                key={preset.key}
                type="button"
                className={`column-preset-btn ${hasExactColumnKeys(Array.from(selected), keys) ? 'active' : ''}`}
                onClick={() => onPreset ? onPreset(preset) : onChange(keys)}
                title={preset.title || `Show ${preset.label} columns`}
              >
                {preset.label}
              </button>
            );
          })}
        </div>
      )}
      <div className="column-picker-options">
        {columns.filter((col) => !col.always).map((col) => {
          const disabled = required.has(col.key);
          return (
            <label key={col.key} title={col.tooltip || col.title} className="column-picker-option">
              <input
                type="checkbox"
                checked={selected.has(col.key)}
                disabled={disabled}
                onChange={(event) => {
                  const next = new Set(selected);
                  if (event.target.checked) next.add(col.key);
                  else next.delete(col.key);
                  onChange(normalizeColumnKeys(columns, Array.from(next), requiredKeys));
                }}
              />
              <span>{col.label}</span>
            </label>
          );
        })}
      </div>
      <button type="button" onClick={onReset} className="column-picker-reset">
        Reset to preset
      </button>
    </div>
  );
}

export function ColumnPickerButton({ open, onClick, label = 'Columns' }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`column-picker-trigger ${open ? 'active' : ''}`}
      aria-expanded={open}
    >
      {label}
    </button>
  );
}
