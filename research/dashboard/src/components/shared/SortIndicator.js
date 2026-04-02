import React from 'react';

export default function SortIndicator({ active, desc }) {
  if (!active) return null;
  return (
    <span style={{ marginLeft: 4, fontSize: 10 }}>
      {desc ? '\u25BC' : '\u25B2'}
    </span>
  );
}
