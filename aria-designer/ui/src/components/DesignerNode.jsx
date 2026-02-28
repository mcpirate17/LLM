import { memo } from 'react'
import { Handle, Position } from '@xyflow/react'
import {
  HelpCircle,
  CheckCircle2,
  Loader,
  XCircle,
  Box,
} from 'lucide-react'
import { CATEGORY_ICONS } from '../utils/categoryConfig'

function formatFlops(n) {
  if (n == null) return ''
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'G'
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M'
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K'
  return String(n)
}

const DTYPE_COLORS = {
  tensor: '#17a3ff',
  scalar: '#f0a020',
  index: '#a060ff',
  mask: '#ff6060',
  complex_tensor: '#20d0a0',
}

function DesignerNode({ data, selected, onHelp }) {
  const { label, category, inputs = [], outputs = [], params = {}, performance = {} } = data
  const IconComponent = CATEGORY_ICONS[category] || Box
  const evalStatus = data.evalStatus // 'running' | 'pass' | 'fail' | null
  const evalClass = evalStatus ? `eval-${evalStatus}` : ''

  return (
    <div className={`designer-node cat-${category || 'default'} ${selected ? 'selected' : ''} ${evalClass}`}>
      {inputs.map((port, i) => (
        <Handle
          key={`in-${port.name}`}
          type="target"
          position={Position.Top}
          id={port.name}
          style={{
            left: inputs.length === 1 ? '50%' : `${((i + 1) / (inputs.length + 1)) * 100}%`,
            background: DTYPE_COLORS[port.dtype] || '#888',
          }}
          title={`${port.name} (${port.dtype})`}
        />
      ))}

      <div className="node-header">
        <div className="node-header-row">
          <div className="node-header-left">
            <span className="node-icon">
              <IconComponent size={14} />
            </span>
            <span className="node-label">{label}</span>
          </div>
          <button 
            className="node-help-btn" 
            onClick={(e) => {
              e.stopPropagation()
              if (typeof onHelp === 'function') {
                onHelp()
              }
            }}
            title="Open component help in Properties panel"
          >
            <HelpCircle size={12} />
          </button>
        </div>
        <span className="node-cat">{category}</span>
      </div>

      {performance.has_params && (
        <div className="node-badge">{performance.param_formula || 'params'}</div>
      )}

      {data.preview && (
        <div className="node-preview">
          {data.preview.shape && <div>Shape: [{data.preview.shape.join(', ')}]</div>}
          {data.preview.mean !== undefined && <div>&mu;: {data.preview.mean.toFixed(2)}</div>}
        </div>
      )}

      {data.profile && (
        <div className="node-profile-overlay">
          <span>{formatFlops(data.profile.flops)}</span>
          {data.profile.has_native_kernel && <span className="native-badge">C</span>}
        </div>
      )}

      {data.routing && (
        <div className="node-routing-overlay" title="Routing efficiency & telemetry">
          {data.routing.savings_ratio != null && (
            <span className="routing-savings">-{Math.round(data.routing.savings_ratio * 100)}% tokens</span>
          )}
          {data.routing.heatmap && (
            <div className="routing-heatmap-mini">
              {data.routing.heatmap.slice(0, 10).map((v, i) => (
                <div 
                  key={i} 
                  style={{ 
                    height: 4, 
                    flex: 1, 
                    background: `rgba(0, 255, 128, ${v})`,
                    borderRadius: 1
                  }} 
                />
              ))}
            </div>
          )}
        </div>
      )}

      {data.compression && data.compression.ratio != null && (
        <div className="node-compression-overlay" title="Parameter compression ratio">
          <span className="compression-badge">
            {data.compression.ratio < 1.0 ? `${(1/data.compression.ratio).toFixed(1)}x compressed` : 'dense'}
          </span>
        </div>
      )}

      {evalStatus === 'running' && (
        <div className="node-eval-badge running">
          <Loader size={10} /> evaluating...
        </div>
      )}
      {evalStatus === 'pass' && (
        <div className="node-eval-badge pass">
          <CheckCircle2 size={10} /> passed
        </div>
      )}
      {evalStatus === 'fail' && (
        <div className="node-eval-badge fail">
          <XCircle size={10} /> failed
        </div>
      )}
      {data.evalError && (
        <div className="node-eval-error">{data.evalError}</div>
      )}

      {outputs.map((port, i) => (
        <Handle
          key={`out-${port.name}`}
          type="source"
          position={Position.Bottom}
          id={port.name}
          style={{
            left: outputs.length === 1 ? '50%' : `${((i + 1) / (outputs.length + 1)) * 100}%`,
            background: DTYPE_COLORS[port.dtype] || '#888',
          }}
          title={`${port.name} (${port.dtype})`}
        />
      ))}
    </div>
  )
}

export default memo(DesignerNode)
