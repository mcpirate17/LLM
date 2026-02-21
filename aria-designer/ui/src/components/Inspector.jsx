import { memo, useCallback, useEffect, useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import {
  ArrowUpDown, Sigma, Grid3X3, Puzzle, Waves, Activity,
  FunctionSquare, Shuffle, Layers, GitFork, Scale, Ruler,
  Hexagon, Box, Compass, Database, Filter, Repeat,
  ChevronDown, ChevronRight, AlertCircle,
} from 'lucide-react'

const CATEGORY_ICONS = {
  io: ArrowUpDown, math: Sigma, linear_algebra: Grid3X3,
  structural: Puzzle, sequence: Waves, frequency: Activity,
  functional: FunctionSquare, mixing: Shuffle, channel_mixing: Layers,
  routing: GitFork, normalization: Scale, positional: Ruler,
  representation: Hexagon, topology: Compass, blocks: Box,
  math_space: Compass, data_io: Database, data_transform: Filter,
  control_flow: Repeat,
}

const CATEGORY_COLORS = {
  io: '#17a3ff', math: '#24d1a0', linear_algebra: '#a060ff',
  structural: '#f0a020', sequence: '#ff6090', frequency: '#20c0f0',
  functional: '#c060c0', mixing: '#ff8040', channel_mixing: '#e0c040',
  routing: '#60c060', normalization: '#8090ff', positional: '#ff60a0',
  representation: '#40d0d0', topology: '#d09040', blocks: '#90c0ff',
  math_space: '#2bd9a9', data_io: '#1aa5ff', data_transform: '#ff9a3d',
  control_flow: '#8bdc65',
}

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

function Inspector({ selectedNode, allComponents, nodeCount, edgeCount, onParamChange, helpRequest }) {
  const comp = selectedNode?.data
  const manifest = comp?.manifest
  const [showHelp, setShowHelp] = useState(false)
  const [showPorts, setShowPorts] = useState(true)
  const [showPerf, setShowPerf] = useState(false)

  const handleParamChange = useCallback((paramName, value) => {
    if (onParamChange && selectedNode) {
      onParamChange(selectedNode.id, paramName, value)
    }
  }, [onParamChange, selectedNode])

  useEffect(() => {
    if (!selectedNode || !helpRequest) return
    if (helpRequest.nodeId === selectedNode.id) {
      setShowHelp(true)
    }
  }, [helpRequest, selectedNode])

  const params = useMemo(() => {
    if (!comp) return {}
    const baseParams = { ...(manifest?.params || {}) }
    const componentId = String(comp?.componentId || '').toLowerCase()

    if (componentId.endsWith('/input') || componentId === 'input') {
      baseParams.source_type = baseParams.source_type || {
        type: 'enum',
        default: 'random',
        options: ['random', 'synthetic', 'binary', 'file'],
        description: 'How this input node generates or loads its input data.',
      }
      baseParams.batch_size = baseParams.batch_size || {
        type: 'integer',
        default: 2,
        constraints: { min: 1, max: 8192 },
        description: 'Batch dimension B.',
      }
      baseParams.seq_len = baseParams.seq_len || {
        type: 'integer',
        default: 16,
        constraints: { min: 1, max: 65536 },
        description: 'Sequence length S.',
      }
      baseParams.seed = baseParams.seed || {
        type: 'integer',
        default: 42,
        constraints: { min: 0, max: 2147483647 },
        description: 'Random seed for reproducible random/synthetic inputs.',
      }

      const sourceType = comp?.paramValues?.source_type || baseParams.source_type.default
      if (sourceType === 'synthetic') {
        baseParams.synthetic_pattern = baseParams.synthetic_pattern || {
          type: 'enum',
          default: 'gaussian',
          options: ['gaussian', 'uniform', 'sine', 'sawtooth', 'impulse'],
          description: 'Pattern used to generate synthetic tensor values.',
        }
      }
      if (sourceType === 'binary') {
        baseParams.binary_path = baseParams.binary_path || {
          type: 'string',
          default: '',
          format: 'file',
          description: 'Path to binary input file.',
        }
      }
      if (sourceType === 'file') {
        baseParams.file_path = baseParams.file_path || {
          type: 'string',
          default: '',
          format: 'file',
          description: 'Path to a file source (CSV/NPY/JSON as supported by runtime).',
        }
      }
    }

    return baseParams
  }, [manifest?.params, comp?.componentId, comp?.paramValues])

  if (!selectedNode) {
    return (
      <div className="inspector-panel">
        <p className="muted">Select a node to view and edit its properties.</p>
        <div className="status-section">
          <h3>Workflow</h3>
          <div className="status-grid">
            <div className="stat"><span className="stat-val">{nodeCount}</span><span className="stat-label">nodes</span></div>
            <div className="stat"><span className="stat-val">{edgeCount}</span><span className="stat-label">edges</span></div>
          </div>
        </div>
      </div>
    )
  }

  const IconComponent = CATEGORY_ICONS[comp?.category] || Box
  const catColor = CATEGORY_COLORS[comp?.category] || '#888'
  const hasParams = Object.keys(params).length > 0

  return (
    <div className="inspector-panel">
      {/* Header with icon, name, category */}
      <div className="props-header" style={{ borderLeftColor: catColor }}>
        <div className="props-header-row">
          <span className="props-icon" style={{ color: catColor }}>
            {IconComponent ? <IconComponent size={20} /> : <Box size={20} />}
          </span>
          <div>
            <div className="props-name">{comp?.label || 'Unknown'}</div>
            <div className="props-cat" style={{ color: catColor }}>{comp?.category || 'other'}</div>
          </div>
        </div>
        <div className="props-id">{selectedNode?.id}</div>
      </div>

      {/* Description */}
      {comp.description && (
        <div className="props-desc">{comp.description}</div>
      )}

      {/* Validation errors from node highlighting */}
      {comp.errors?.length > 0 && (
        <div className="props-errors">
          {comp.errors.map((err, i) => (
            <div key={i} className="props-error-item">
              <AlertCircle size={12} /> {err}
            </div>
          ))}
        </div>
      )}

      {/* Parameters — the main configuration section */}
      {hasParams && (
        <div className="props-section">
          <h4>Configuration</h4>
          {Object.entries(params).map(([key, schema]) => {
            const value = comp.paramValues?.[key]
            const isEmpty = value === undefined || value === null || value === ''
            const isRequired = schema.required === true
            const requiredError = isRequired && isEmpty
            const backendErrors = comp.paramErrors?.[key] || []
            const hasError = requiredError || backendErrors.length > 0

            return (
              <div key={key} className={`prop-field ${hasError ? 'field-error' : ''}`}>
                <div className="prop-field-header">
                  <label>{key}</label>
                  {isRequired && <span className="required-dot" title="Required">*</span>}
                </div>
                {schema.description && (
                  <div className="prop-field-desc">{schema.description}</div>
                )}
                <ParamInput
                  name={key}
                  schema={schema}
                  value={value ?? schema.default}
                  paramValues={comp.paramValues || {}}
                  onChange={(v) => handleParamChange(key, v)}
                />
                {requiredError && (
                  <div className="field-error-msg">
                    <AlertCircle size={11} /> Required
                  </div>
                )}
                {backendErrors.map((msg, i) => (
                  <div key={`${key}-err-${i}`} className="field-error-msg">
                    <AlertCircle size={11} /> {msg}
                  </div>
                ))}
              </div>
            )
          })}
        </div>
      )}
      {!hasParams && (
        <div className="props-section">
          <h4>Configuration</h4>
          <div className="muted">This component has no configurable parameters yet.</div>
        </div>
      )}

      {/* Ports — collapsible */}
      {(manifest?.inputs?.length > 0 || manifest?.outputs?.length > 0) && (
        <div className="props-section">
          <button className="section-toggle" onClick={() => setShowPorts(!showPorts)}>
            {showPorts ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            <h4>Ports</h4>
          </button>
          {showPorts && (
            <>
              {manifest.inputs?.length > 0 && (
                <div className="port-group">
                  <div className="port-group-label">Inputs</div>
                  {manifest.inputs.map((p) => (
                    <div key={p.name} className="port-row">
                      <span className="port-name">{p.name}</span>
                      <span className="port-dtype">{p.dtype}</span>
                      {p.shape && <span className="port-shape">[{p.shape.join(', ')}]</span>}
                    </div>
                  ))}
                </div>
              )}
              {manifest.outputs?.length > 0 && (
                <div className="port-group">
                  <div className="port-group-label">Outputs</div>
                  {manifest.outputs.map((p) => (
                    <div key={p.name} className="port-row">
                      <span className="port-name">{p.name}</span>
                      <span className="port-dtype">{p.dtype}</span>
                      {p.shape && <span className="port-shape">[{p.shape.join(', ')}]</span>}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Shape preview from run results */}
      {comp.preview && (
        <div className="props-section">
          <h4>Shape Preview</h4>
          <div className="shape-preview">
            {comp.preview.shape && <div>Output: [{comp.preview.shape.join(', ')}]</div>}
            {comp.preview.mean !== undefined && <div>Mean: {comp.preview.mean.toFixed(4)}</div>}
            {comp.preview.std !== undefined && <div>Std: {comp.preview.std.toFixed(4)}</div>}
          </div>
        </div>
      )}

      {/* Performance — collapsible */}
      {manifest?.performance && (
        <div className="props-section">
          <button className="section-toggle" onClick={() => setShowPerf(!showPerf)}>
            {showPerf ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            <h4>Performance</h4>
          </button>
          {showPerf && (
            <div className="perf-info">
              {manifest.performance.has_params && (
                <div className="perf-row">
                  <span className="perf-label">Params</span>
                  <span className="perf-value">{manifest.performance.param_formula}</span>
                </div>
              )}
              {manifest.performance.flops_formula && (
                <div className="perf-row">
                  <span className="perf-label">FLOPs</span>
                  <span className="perf-value">{manifest.performance.flops_formula}</span>
                </div>
              )}
              {manifest.performance.numerically_risky && (
                <div className="perf-row warn">Numerically risky</div>
              )}
              {manifest.performance.preserves_gradient === false && (
                <div className="perf-row warn">May block gradients</div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Help — collapsible "Learn more" */}
      {comp.help_md && (
        <div className="props-section">
          <button className="section-toggle" onClick={() => setShowHelp(!showHelp)}>
            {showHelp ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            <h4>Learn More</h4>
          </button>
          {showHelp && (
            <ReactMarkdown className="help-md">{comp.help_md}</ReactMarkdown>
          )}
        </div>
      )}

      {/* Workflow stats */}
      <div className="status-section">
        <div className="status-grid">
          <div className="stat"><span className="stat-val">{nodeCount}</span><span className="stat-label">nodes</span></div>
          <div className="stat"><span className="stat-val">{edgeCount}</span><span className="stat-label">edges</span></div>
        </div>
      </div>
    </div>
  )
}

function ParamInput({ name, schema, value, paramValues, onChange }) {
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
              return
            }
            try {
              const parsed = JSON.parse(trimmed)
              onChange(parsed)
            } catch {}
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
        />
        <input
          type="file"
          {...(schema.format === 'directory' ? { webkitdirectory: 'true', directory: 'true' } : {})}
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (!file) return
            onChange(file.name)
          }}
        />
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
    />
  )
}

export default memo(Inspector)
