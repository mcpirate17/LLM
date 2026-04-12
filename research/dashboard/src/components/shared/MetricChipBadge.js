import React from 'react';
import { reliabilityColor } from '../../utils/colors';

function MetricChipBadge({ chip }) {
  const color = reliabilityColor(chip.reliability);
  return (
    <span
      title={`${chip.label}: ${chip.source}, ${chip.reliability} reliability`}
      style={{
        fontSize: 10,
        padding: '1px 5px',
        borderRadius: 4,
        border: `1px solid ${color}55`,
        color,
        background: `${color}22`,
        whiteSpace: 'nowrap',
      }}
    >
      {chip.label}: {chip.source}
    </span>
  );
}

export function MetricChipList({ chips, maxWidth = 220, wrap = true }) {
  if (!chips?.length) return null;
  return (
    <div
      style={{
        display: 'flex',
        gap: 4,
        flexWrap: wrap ? 'wrap' : 'nowrap',
        maxWidth,
        overflow: 'hidden',
        whiteSpace: wrap ? 'normal' : 'nowrap',
      }}
    >
      {chips.map((chip, i) => (
        <MetricChipBadge key={chip.label || i} chip={chip} />
      ))}
    </div>
  );
}

export default React.memo(MetricChipBadge);
