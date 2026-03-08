import React from 'react';
import { AlertCircle, CheckCircle2, RotateCcw } from 'lucide-react';
import ParamInput from './ParamInput';

function normalizeNumberValue(value, schemaType) {
  if (value === null || value === undefined || value === '') return null;
  const num = schemaType === 'integer' ? parseInt(value, 10) : parseFloat(value);
  return Number.isNaN(num) ? null : num;
}

function validateParamValue(name, schema, value) {
  const errors = [];
  const isEmpty = value === undefined || value === null || value === '';
  if (schema?.required === true && isEmpty) {
    errors.push('Required');
    return errors;
  }
  if (isEmpty) return errors;

  if (schema?.type === 'integer' || schema?.type === 'float') {
    const parsed = normalizeNumberValue(value, schema.type);
    if (parsed === null) {
      errors.push('Must be a valid number');
      return errors;
    }
    const min = schema?.constraints?.min;
    const max = schema?.constraints?.max;
    if (typeof min === 'number' && parsed < min) errors.push(`Must be >= ${min}`);
    if (typeof max === 'number' && parsed > max) errors.push(`Must be <= ${max}`);
  }

  const isJsonField = schema?.type === 'json' || schema?.type === 'object' || schema?.format === 'json';
  if (isJsonField && typeof value === 'string' && value.trim()) {
    try {
      JSON.parse(value);
    } catch {
      errors.push('Invalid JSON');
    }
  }

  if (schema?.type === 'enum' && Array.isArray(schema.options) && schema.options.length > 0) {
    if (Array.isArray(value)) {
      const invalid = value.find((v) => !schema.options.includes(v));
      if (invalid !== undefined) errors.push(`Invalid option: ${invalid}`);
    } else if (!schema.options.includes(value)) {
      errors.push('Invalid option selected');
    }
  }

  return errors;
}

function isAtDefaultValue(rawValue, schema) {
  if (!schema || !Object.prototype.hasOwnProperty.call(schema, 'default')) return rawValue === undefined;
  const activeValue = rawValue === undefined ? schema.default : rawValue;
  try {
    return JSON.stringify(activeValue) === JSON.stringify(schema.default);
  } catch {
    return activeValue === schema.default;
  }
}

const PropertyField = ({ 
  paramKey, 
  schema, 
  value, 
  paramErrors, 
  touched, 
  onParamChange, 
  onReset,
  quickPresets,
  paramValues
}) => {
  const isRequired = schema.required === true;
  const localErrors = validateParamValue(paramKey, schema, value);
  const backendErrors = paramErrors || [];
  const allErrors = [...localErrors, ...backendErrors];
  const hasError = allErrors.length > 0;
  const hasValue = value !== undefined && value !== null && value !== '';
  const isValid = touched && !hasError && (hasValue || !isRequired);
  const canReset = !isAtDefaultValue(value, schema);

  return (
    <div className={`prop-field ${hasError ? 'field-error' : ''} ${isValid ? 'field-valid' : ''}`}>
      <div className="prop-field-header">
        <label>{paramKey}</label>
        <div className="prop-field-actions">
          {isRequired && <span className="required-dot" title="Required">*</span>}
          <button
            type="button"
            className="field-reset-btn"
            title={`Reset ${paramKey} to default`}
            onClick={onReset}
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
              key={`${paramKey}-${preset.label}`}
              type="button"
              className="param-preset-chip"
              onClick={() => onParamChange(preset.value)}
              title={`Set ${paramKey} to ${String(preset.value)}`}
            >
              {preset.label}
            </button>
          ))}
        </div>
      )}
      <ParamInput
        name={paramKey}
        schema={schema}
        value={value ?? schema.default}
        paramValues={paramValues}
        onChange={onParamChange}
        onBlur={() => {}}
      />
      {allErrors.map((msg, i) => (
        <div key={`${paramKey}-err-${i}`} className="field-error-msg">
          <AlertCircle size={11} /> {msg}
        </div>
      ))}
      {isValid && (
        <div className="field-valid-msg">
          <CheckCircle2 size={11} /> Looks good
        </div>
      )}
    </div>
  );
};

export default PropertyField;
