import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import {
  ChevronDown, ChevronRight, AlertCircle, Box, RotateCcw, CheckCircle2,
} from 'lucide-react'
import { CATEGORY_ICONS, CATEGORY_COLORS } from '../utils/categoryConfig'
import ParamInput from './ParamInput'

function getParamGroup(paramName, schema) {
  const key = String(paramName || '').toLowerCase()
  if (/source|file|path|schema|column|dataset|binary|synthetic/.test(key)) return 'Data Source'
  if (/batch|seq|token|dim|channel|head|width|height|size|length/.test(key)) return 'Shape & Size'
  if (/dropout|lr|learning|decay|epsilon|eps|momentum|beta|alpha|regular/.test(key)) return 'Training & Regularization'
  if (schema?.type === 'json' || schema?.type === 'object' || schema?.format === 'json') return 'Advanced'
  if (/mode|strategy|scope|policy|backend|dtype|device/.test(key)) return 'Advanced'
  return 'Core'
}

function normalizeNumberValue(value, schemaType) {
  if (value === null || value === undefined || value === '') return null
  const num = schemaType === 'integer' ? parseInt(value, 10) : parseFloat(value)
  return Number.isNaN(num) ? null : num
}

function validateParamValue(name, schema, value) {
  const errors = []
  const isEmpty = value === undefined || value === null || value === ''
  if (schema?.required === true && isEmpty) {
    errors.push('Required')
    return errors
  }
  if (isEmpty) return errors

  if (schema?.type === 'integer' || schema?.type === 'float') {
    const parsed = normalizeNumberValue(value, schema.type)
    if (parsed === null) {
      errors.push('Must be a valid number')
      return errors
    }
    const min = schema?.constraints?.min
    const max = schema?.constraints?.max
    if (typeof min === 'number' && parsed < min) errors.push(`Must be >= ${min}`)
    if (typeof max === 'number' && parsed > max) errors.push(`Must be <= ${max}`)
  }

  const isJsonField = schema?.type === 'json' || schema?.type === 'object' || schema?.format === 'json'
  if (isJsonField && typeof value === 'string' && value.trim()) {
    try {
      JSON.parse(value)
    } catch {
      errors.push('Invalid JSON')
    }
  }

  if (schema?.type === 'enum' && Array.isArray(schema.options) && schema.options.length > 0) {
    if (Array.isArray(value)) {
      const invalid = value.find((v) => !schema.options.includes(v))
      if (invalid !== undefined) errors.push(`Invalid option: ${invalid}`)
    } else if (!schema.options.includes(value)) {
      errors.push('Invalid option selected')
    }
  }

  return errors
}

function isAtDefaultValue(rawValue, schema) {
  if (!schema || !Object.prototype.hasOwnProperty.call(schema, 'default')) return rawValue === undefined
  const activeValue = rawValue === undefined ? schema.default : rawValue
  try {
    return JSON.stringify(activeValue) === JSON.stringify(schema.default)
  } catch {
    return activeValue === schema.default
  }
}

function getQuickPresets({ name, schema, componentId }) {
  const key = String(name || '').toLowerCase()
  const comp = String(componentId || '').toLowerCase()
  if (key === 'source_type') {
    return [
      { label: 'Random', value: 'random' },
      { label: 'Synthetic', value: 'synthetic' },
      { label: 'File', value: 'file' },
    ]
  }
  if (key === 'synthetic_pattern') {
    return [
      { label: 'Gaussian', value: 'gaussian' },
      { label: 'Uniform', value: 'uniform' },
      { label: 'Sine', value: 'sine' },
    ]
  }
  if (key === 'batch_size') {
    return [
      { label: 'B=1', value: 1 },
      { label: 'B=8', value: 8 },
      { label: 'B=32', value: 32 },
    ]
  }
  if (key === 'seq_len') {
    return [
      { label: 'S=16', value: 16 },
      { label: 'S=128', value: 128 },
      { label: 'S=512', value: 512 },
    ]
  }
  if (key === 'dropout') {
    return [
      { label: 'Off', value: 0.0 },
      { label: 'Light', value: 0.1 },
      { label: 'Strong', value: 0.3 },
    ]
  }
  if (key === 'seed') {
    return [
      { label: '42', value: 42 },
      { label: '123', value: 123 },
      { label: '2025', value: 2025 },
    ]
  }
  if (key === 'split_scope') {
    return [
      { label: 'Features', value: 'feature' },
      { label: 'Tokens', value: 'token' },
    ]
  }
  if (key === 'filter_scope') {
    return [
      { label: 'Rows', value: 'row' },
      { label: 'Tokens', value: 'token' },
      { label: 'Features', value: 'feature' },
    ]
  }
  if (schema?.type === 'boolean') {
    return [
      { label: 'On', value: true },
      { label: 'Off', value: false },
    ]
  }
  if (comp.endsWith('/input') && key === 'file_path') {
    return [
      { label: 'Use sample', value: 'data/sample.npy' },
      { label: 'Use CSV', value: 'data/sample.csv' },
    ]
  }
  return []
}

function Inspector({ selectedNode, allComponents, nodeCount, edgeCount, onParamChange, helpRequest }) {
  const comp = selectedNode?.data
  const manifest = comp?.manifest
  const [showHelp, setShowHelp] = useState(false)
  const [showPorts, setShowPorts] = useState(true)
  const [showPerf, setShowPerf] = useState(false)
  const [touchedFields, setTouchedFields] = useState({})
  const [collapsedGroups, setCollapsedGroups] = useState({})
  const groupRefs = useRef({})

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

  const IconComponent = CATEGORY_ICONS[comp?.category] || Box
  const catColor = CATEGORY_COLORS[comp?.category] || '#888'
  const hasParams = Object.keys(params).length > 0
  const groupedParams = useMemo(() => {
    const orderedGroups = ['Core', 'Data Source', 'Shape & Size', 'Training & Regularization', 'Advanced']
    const groups = new Map(orderedGroups.map((name) => [name, []]))
    Object.entries(params).forEach(([key, schema]) => {
      const group = getParamGroup(key, schema)
      if (!groups.has(group)) groups.set(group, [])
      groups.get(group).push([key, schema])
    })
    return Array.from(groups.entries()).filter(([, entries]) => entries.length > 0)
  }, [params])
  const hasAnyDefault = useMemo(
    () => Object.values(params).some((schema) => Object.prototype.hasOwnProperty.call(schema, 'default')),
    [params]
  )
  const hasModifiedValues = useMemo(
    () => Object.entries(params).some(([key, schema]) => !isAtDefaultValue(comp.paramValues?.[key], schema)),
    [params, comp?.paramValues]
  )
  useEffect(() => {
    setCollapsedGroups((prev) => {
      const next = { ...prev }
      groupedParams.forEach(([groupName]) => {
        if (!(groupName in next)) next[groupName] = false
      })
      return next
    })
  }, [groupedParams])

  const resetFieldToDefault = useCallback((paramKey, schema) => {
    const hasDefault = Object.prototype.hasOwnProperty.call(schema, 'default')
    handleParamChange(paramKey, hasDefault ? schema.default : null)
    setTouchedFields((prev) => ({ ...prev, [paramKey]: true }))
  }, [handleParamChange])

  const resetAllToDefaults = useCallback(() => {
    Object.entries(params).forEach(([key, schema]) => {
      const hasDefault = Object.prototype.hasOwnProperty.call(schema, 'default')
      handleParamChange(key, hasDefault ? schema.default : null)
    })
    const nextTouched = {}
    Object.keys(params).forEach((key) => { nextTouched[key] = true })
    setTouchedFields(nextTouched)
  }, [handleParamChange, params])

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
          <div className="props-section-head">
            <h4>Configuration</h4>
            <button
              type="button"
              className="field-reset-btn"
              onClick={resetAllToDefaults}
              disabled={!hasAnyDefault || !hasModifiedValues}
              title="Reset all configuration fields to defaults"
            >
              <RotateCcw size={12} />
              Reset all
            </button>
          </div>
          <div className="param-nav" role="tablist" aria-label="Configuration groups">
            {groupedParams.map(([groupName]) => (
              <button
                key={`nav-${groupName}`}
                type="button"
                className={`param-nav-chip ${collapsedGroups[groupName] ? 'is-collapsed' : ''}`}
                onClick={() => {
                  const target = groupRefs.current[groupName]
                  if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' })
                  setCollapsedGroups((prev) => ({ ...prev, [groupName]: false }))
                }}
                title={`Jump to ${groupName}`}
              >
                {groupName}
              </button>
            ))}
          </div>
          {groupedParams.map(([groupName, entries]) => (
            <div
              key={groupName}
              className="param-group"
              ref={(el) => { groupRefs.current[groupName] = el }}
            >
              <button
                type="button"
                className="param-group-toggle"
                aria-expanded={!collapsedGroups[groupName]}
                onClick={() => setCollapsedGroups((prev) => ({ ...prev, [groupName]: !prev[groupName] }))}
              >
                {collapsedGroups[groupName] ? <ChevronRight size={13} /> : <ChevronDown size={13} />}
                <span className="param-group-title">{groupName}</span>
                <span className="param-group-count">{entries.length}</span>
              </button>
              {!collapsedGroups[groupName] && entries.map(([key, schema]) => {
                const value = comp.paramValues?.[key]
                const isRequired = schema.required === true
                const localErrors = validateParamValue(key, schema, value)
                const backendErrors = comp.paramErrors?.[key] || []
                const allErrors = [...localErrors, ...backendErrors]
                const hasError = allErrors.length > 0
                const touched = touchedFields[key] === true
                const hasValue = value !== undefined && value !== null && value !== ''
                const isValid = touched && !hasError && (hasValue || !isRequired)
                const canReset = !isAtDefaultValue(value, schema)
                const quickPresets = getQuickPresets({ name: key, schema, componentId: comp?.componentId })

                return (
                  <div key={key} className={`prop-field ${hasError ? 'field-error' : ''} ${isValid ? 'field-valid' : ''}`}>
                    <div className="prop-field-header">
                      <label>{key}</label>
                      <div className="prop-field-actions">
                        {isRequired && <span className="required-dot" title="Required">*</span>}
                        <button
                          type="button"
                          className="field-reset-btn"
                          title={`Reset ${key} to default`}
                          onClick={() => resetFieldToDefault(key, schema)}
                          disabled={!canReset}
                        >
                          <RotateCcw size={11} />
                          Reset
                        </button>
                      </div>
                    </div>
                    {schema.description && (
                      <div className="prop-field-desc">{schema.description}</div>
                    )}
                    {quickPresets.length > 0 && (
                      <div className="param-preset-row">
                        {quickPresets.map((preset) => (
                          <button
                            key={`${key}-${preset.label}`}
                            type="button"
                            className="param-preset-chip"
                            onClick={() => {
                              setTouchedFields((prev) => ({ ...prev, [key]: true }))
                              handleParamChange(key, preset.value)
                            }}
                            title={`Set ${key} to ${String(preset.value)}`}
                          >
                            {preset.label}
                          </button>
                        ))}
                      </div>
                    )}
                    <ParamInput
                      name={key}
                      schema={schema}
                      value={value ?? schema.default}
                      paramValues={comp.paramValues || {}}
                      onChange={(v) => {
                        setTouchedFields((prev) => ({ ...prev, [key]: true }))
                        handleParamChange(key, v)
                      }}
                      onBlur={() => {
                        setTouchedFields((prev) => ({ ...prev, [key]: true }))
                      }}
                    />
                    {allErrors.map((msg, i) => (
                      <div key={`${key}-err-${i}`} className="field-error-msg">
                        <AlertCircle size={11} /> {msg}
                      </div>
                    ))}
                    {isValid && (
                      <div className="field-valid-msg">
                        <CheckCircle2 size={11} /> Looks good
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          ))}
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
          <button type="button" className="section-toggle" aria-expanded={showPorts} onClick={() => setShowPorts(!showPorts)}>
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
          <button type="button" className="section-toggle" aria-expanded={showPerf} onClick={() => setShowPerf(!showPerf)}>
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
          <button type="button" className="section-toggle" aria-expanded={showHelp} onClick={() => setShowHelp(!showHelp)}>
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

export default memo(Inspector)
