import React from 'react';
import { Box } from 'lucide-react';
import { CATEGORY_ICONS, CATEGORY_COLORS } from '../../utils/categoryConfig';

const InspectorHeader = ({ comp, nodeId }) => {
  const IconComponent = CATEGORY_ICONS[comp?.category] || Box;
  const catColor = CATEGORY_COLORS[comp?.category] || '#888';

  return (
    <div className="props-header" style={{ borderLeftColor: catColor }}>
      <div className="props-header-row">
        <span className="props-icon" style={{ color: catColor }}>
          <IconComponent size={20} />
        </span>
        <div>
          <div className="props-name">{comp?.label || 'Unknown'}</div>
          <div className="props-cat" style={{ color: catColor }}>{comp?.category || 'other'}</div>
        </div>
      </div>
      <div className="props-id">{nodeId}</div>
    </div>
  );
};

export default InspectorHeader;
