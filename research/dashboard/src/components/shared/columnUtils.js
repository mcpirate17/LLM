export function normalizeColumnKeys(columns, keys, requiredKeys = []) {
  const valid = new Set(columns.map((col) => col.key));
  const always = columns.filter((col) => col.always).map((col) => col.key);
  const normalized = Array.isArray(keys) ? keys.filter((key) => valid.has(key)) : [];
  const required = Array.isArray(requiredKeys) ? requiredKeys.filter((key) => valid.has(key)) : [];
  return Array.from(new Set([...always, ...required, ...normalized]));
}

export function presetColumns(columns, presets, viewKey) {
  const preset = presets.find((item) => item.key === viewKey) || presets[0];
  const allowed = new Set(normalizeColumnKeys(columns, preset.columns || []));
  return columns.filter((col) => allowed.has(col.key));
}

export function visibleColumns(columns, presets, viewKey, customKeys, requiredKeys = []) {
  if (Array.isArray(customKeys) && customKeys.length > 0) {
    const allowed = new Set(normalizeColumnKeys(columns, customKeys, requiredKeys));
    return columns.filter((col) => allowed.has(col.key));
  }
  return presetColumns(columns, presets, viewKey);
}

export function selectedColumnKeys(columns, presets, viewKey, customKeys, requiredKeys = []) {
  if (Array.isArray(customKeys) && customKeys.length > 0) {
    return normalizeColumnKeys(columns, customKeys, requiredKeys);
  }
  return presetColumns(columns, presets, viewKey).map((col) => col.key);
}

export function groupedSpans(columns, groups) {
  return groups
    .map((group) => ({
      ...group,
      span: columns.filter((col) => col.group === group.key && !col.sticky).length,
    }))
    .filter((group) => group.span > 0);
}

export function tableMinWidth(columns, fallbackWidth = 92) {
  return columns.reduce((total, col) => total + (col.width || fallbackWidth), 0);
}

export function hasExactColumnKeys(activeKeys, targetKeys) {
  if (!Array.isArray(activeKeys) || !Array.isArray(targetKeys)) return false;
  if (activeKeys.length !== targetKeys.length) return false;
  const keySet = new Set(targetKeys);
  return activeKeys.every((key) => keySet.has(key));
}
