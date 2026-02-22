export function normalizeFilterValue(value) {
  if (value === null || value === undefined) return '';
  return String(value).toLowerCase();
}

export function filterRowsByQuery(rows, query, keys) {
  if (!query) return rows;
  const q = normalizeFilterValue(query).trim();
  if (!q) return rows;
  return rows.filter((row) => {
    return keys.some((key) => {
      const value = typeof key === 'function' ? key(row) : row?.[key];
      return normalizeFilterValue(value).includes(q);
    });
  });
}
