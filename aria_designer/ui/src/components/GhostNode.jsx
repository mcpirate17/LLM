import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Sparkles, Plus } from 'lucide-react';
import { CATEGORY_ICONS } from '../utils/categoryConfig';

function GhostNode({ data, selected }) {
  const { label, category, description, reasoning } = data;
  const IconComponent = CATEGORY_ICONS[category] || Sparkles;

  return (
    <div className={`designer-node ghost-node ${selected ? 'selected' : ''}`} title={reasoning || description}>
      <div className="ghost-node-overlay">
        <Sparkles size={16} className="ghost-sparkle" />
      </div>
      
      <div className="node-header">
        <div className="node-header-row">
          <div className="node-header-left">
            <span className="node-icon">
              <IconComponent size={14} />
            </span>
            <span className="node-label">{label}</span>
          </div>
          <div className="ghost-plus-circle">
            <Plus size={12} />
          </div>
        </div>
        <span className="node-cat">ARIA SUGGESTION</span>
      </div>

      <div className="ghost-reasoning">
        {reasoning || 'Aria recommends adding this component here to improve model dynamics.'}
      </div>

      <Handle type="target" position={Position.Top} isConnectable={false} style={{ opacity: 0.5 }} />
      <Handle type="source" position={Position.Bottom} isConnectable={false} style={{ opacity: 0.5 }} />
    </div>
  );
}

export default memo(GhostNode);
