import { useState } from 'react'

const SCOPE_OPTION_LABELS = {
  split_scope: {
    feature: 'By Features (last dimension)',
    token: 'By Tokens (sequence length)',
  },
  filter_scope: {
    row: 'Rows (batch items)',
    token: 'Tokens (sequence positions)',
    feature: 'Features (channels)',
    dataset_row: 'Dataset Rows (records)',
  },
}

const PARAM_SCOPE_HELP = {
  split_scope: 'Choose whether split operates across feature channels or across token positions.',
  filter_scope: 'Choose the level where filtering happens: full rows, individual tokens, or feature channels.',
}

export default function ParamInput({ name, schema, value, paramValues, onChange, onBlur }) {
  const [pickerMeta, setPickerMeta] = useState('')
  const isPathField = schema.type === 'string' && (
    schema.format === 'file' ||
    schema.format === 'path' ||
    schema.format === 'directory' ||
    /path|file|dir/i.test(name)
  )
  const isJsonField = schema.type === 'json' || schema.type === 'object' || schema.format === 'json'
  const isMultiSelectEnum = schema.type === 'enum' && schema.options && (
    schema.multi_select === true || schema.multiple === true || Array.isArray(value)
  )

  if (isJsonField) {
    const renderedValue = typeof value === 'string'
      ? value
      : (value === undefined || value === null || value === ''
        ? ''
        : JSON.stringify(value, null, 2))

    return (
      <div className="json-editor-wrap">
        <textarea
          className="json-editor"
          value={renderedValue}
          rows={6}
          placeholder={schema.default != null ? JSON.stringify(schema.default, null, 2) : '{\n  "key": "value"\n}'}
          onChange={(e) => onChange(e.target.value)}
          onBlur={() => {
            const trimmed = String(renderedValue).trim()
            if (!trimmed) {
              onChange(null)
              onBlur?.()
              return
            }
            try {
              const parsed = JSON.parse(trimmed)
              onChange(parsed)
            } catch {}
            onBlur?.()
          }}
        />
        <div className="param-help">Enter JSON object/array. Value is parsed on blur when valid.</div>
      </div>
    )
  }

  if (isMultiSelectEnum) {
    const rawSelected = Array.isArray(value)
      ? value.map((v) => String(v))
      : String(value ?? '')
          .split(',')
          .map((v) => v.trim())
          .filter(Boolean)
    const selectedSet = new Set(rawSelected)

    const emit = (nextSet) => {
      const values = Array.from(nextSet)
      if (Array.isArray(value) || schema.as_array === true) {
        onChange(values)
      } else {
        onChange(values.join(','))
      }
    }

    return (
      <div className="enum-multiselect-wrap">
        {schema.options.map((opt) => {
          const key = String(opt)
          return (
            <label key={key} className="checkbox-wrap">
              <input
                type="checkbox"
                checked={selectedSet.has(key)}
                onChange={(e) => {
                  const next = new Set(selectedSet)
                  if (e.target.checked) next.add(key)
                  else next.delete(key)
                  emit(next)
                }}
              />
              <span>{key}</span>
            </label>
          )
        })}
      </div>
    )
  }

  if (schema.type === 'string' && schema.format === 'csv_columns') {
    const schemaSourceKey = schema.schema_source_param || 'schema_columns'
    const availableColumns = String(paramValues?.[schemaSourceKey] ?? '')
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)
    const selectedSet = new Set(
      String(value ?? '')
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
    )
    const setSelected = (nextSet) => {
      const merged = Array.from(nextSet)
      onChange(merged.join(','))
    }

    return (
      <div className="csv-columns-editor">
        {availableColumns.length > 0 ? (
          <div className="csv-columns-options">
            {availableColumns.map((col) => (
              <label key={col} className="checkbox-wrap">
                <input
                  type="checkbox"
                  checked={selectedSet.has(col)}
                  onChange={(e) => {
                    const next = new Set(selectedSet)
                    if (e.target.checked) next.add(col)
                    else next.delete(col)
                    setSelected(next)
                  }}
                />
                <span>{col}</span>
              </label>
            ))}
          </div>
        ) : (
          <div className="param-help">Set `schema_columns` first to enable schema-aware column selection.</div>
        )}
        <input
          type="text"
          value={value ?? ''}
          placeholder={schema.default != null ? String(schema.default) : 'col_a,col_b'}
          onChange={(e) => onChange(e.target.value)}
          onBlur={() => onBlur?.()}
        />
      </div>
    )
  }
  if (isPathField) {
    return (
      <div className="file-input-wrap">
        <input
          type="text"
          value={value ?? ''}
          placeholder={schema.default != null ? String(schema.default) : '/path/to/resource'}
          onChange={(e) => onChange(e.target.value)}
          onBlur={() => onBlur?.()}
        />
        <input
          type="file"
          {...(schema.format === 'directory' ? { webkitdirectory: 'true', directory: 'true' } : {})}
          onChange={(e) => {
            const files = Array.from(e.target.files || [])
            const first = files[0]
            if (!first) return

            const inputReportedPath = String(e.target.value || '').replace(/^.*[\\/]/, '')
            let selectedPath = first.webkitRelativePath || inputReportedPath || first.name

            if (schema.format === 'directory' && files.length > 0) {
              const relPath = first.webkitRelativePath || ''
              const rootDir = relPath.includes('/') ? relPath.split('/')[0] : relPath
              selectedPath = rootDir || selectedPath
            }

            onChange(selectedPath)
            setPickerMeta(schema.format === 'directory'
              ? `Picked folder: ${selectedPath}`
              : `Picked file: ${selectedPath}`)
          }}
        />
        <div className="param-help">
          Picker autofills a detected name/path. You can always override manually in the text field.
        </div>
        {pickerMeta && <div className="file-picker-meta">{pickerMeta}</div>}
      </div>
    )
  }
  if (schema.type === 'boolean') {
    return (
      <label className="checkbox-wrap">
        <input
          type="checkbox"
          checked={!!value}
          onChange={(e) => onChange(e.target.checked)}
          onBlur={() => onBlur?.()}
        />
        <span>{value ? 'Yes' : 'No'}</span>
      </label>
    )
  }
  if (schema.type === 'enum' && schema.options) {
    const labels = SCOPE_OPTION_LABELS[name] || {}
    const helper = PARAM_SCOPE_HELP[name]
    return (
      <>
        <select value={value ?? ''} onChange={(e) => onChange(e.target.value)}>
          <option value="" disabled>Select...</option>
          {schema.options.map((opt) => (
            <option key={opt} value={opt} title={labels[opt] || String(opt)}>
              {labels[opt] || opt}
            </option>
          ))}
        </select>
        {helper && <div className="param-help">{helper}</div>}
      </>
    )
  }
  if (schema.type === 'integer' || schema.type === 'float') {
    const min = schema.constraints?.min
    const max = schema.constraints?.max
    const showSlider = min !== undefined && max !== undefined && (max - min) <= 10000
    return (
      <div className="number-input-wrap">
        <input
          type="number"
          value={value ?? ''}
          placeholder={schema.default != null ? String(schema.default) : ''}
          min={min}
          max={max}
          step={schema.type === 'float' ? 0.01 : 1}
          onChange={(e) => {
            const v = e.target.value
            if (v === '') return onChange(null)
            const parsed = schema.type === 'integer' ? parseInt(v, 10) : parseFloat(v)
            onChange(Number.isNaN(parsed) ? null : parsed)
          }}
          onBlur={() => onBlur?.()}
        />
        {showSlider && (
          <input
            type="range"
            className="param-slider"
            value={value ?? schema.default ?? min}
            min={min}
            max={max}
            step={schema.type === 'float' ? 0.01 : 1}
            onChange={(e) => {
              const parsed = schema.type === 'integer'
                ? parseInt(e.target.value, 10)
                : parseFloat(e.target.value)
              onChange(Number.isNaN(parsed) ? null : parsed)
            }}
            onBlur={() => onBlur?.()}
          />
        )}
      </div>
    )
  }
  return (
    <input
      type="text"
      value={value ?? ''}
      placeholder={schema.default != null ? String(schema.default) : ''}
      onChange={(e) => onChange(e.target.value)}
      onBlur={() => onBlur?.()}
    />
  )
}
