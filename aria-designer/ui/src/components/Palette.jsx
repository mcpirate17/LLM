import { memo, useMemo, useState } from 'react'

const CATEGORY_LABELS = {
  // Graph I/O
  io: 'Graph I/O',
  // Data pipeline
  data_io: 'Data Sources & Sinks',
  data_transform: 'Data Transform',
  control_flow: 'Control Flow',
  // Core math
  math: 'Basic Math',
  linear_algebra: 'Linear Layers',
  // Sequence processing
  mixing: 'Sequence Mixing',
  channel_mixing: 'Channel Mixing',
  sequence: 'Sequence Ops',
  frequency: 'Frequency Domain',
  // Architecture
  normalization: 'Normalization',
  positional: 'Positional Encoding',
  structural: 'Structural (Split/Concat)',
  representation: 'Representation',
  // Advanced
  routing: 'Routing & MoE',
  topology: 'Topology',
  blocks: 'Block Templates',
  // Specialized
  functional: 'Functional & Implicit',
  math_space: 'Math Spaces',
}

const CATEGORY_ORDER = [
  // Graph I/O
  'io',
  // Data pipeline
  'data_io', 'data_transform', 'control_flow',
  // Core math
  'math', 'linear_algebra',
  // Sequence processing
  'mixing', 'channel_mixing', 'sequence', 'frequency',
  // Architecture
  'normalization', 'positional', 'structural', 'representation',
  // Advanced
  'routing', 'topology', 'blocks',
  // Specialized
  'functional', 'math_space',
]

function Palette({ components, onDragStart, constraints = {} }) {
  const [search, setSearch] = useState('')
  const [expanded, setExpanded] = useState({})
  const [tooltip, setTooltip] = useState(null)

  const grouped = useMemo(() => {
    const groups = {}
    const q = search.toLowerCase()
    for (const comp of components) {
      if (q && !comp.name.toLowerCase().includes(q) && !comp.id.toLowerCase().includes(q)) continue
      
      const cat = comp.category || 'other'
      
      if (!groups[cat]) groups[cat] = []
      groups[cat].push(comp)
    }
    // Sort categories by predefined order
    const sorted = []
    for (const cat of CATEGORY_ORDER) {
      if (groups[cat]) sorted.push([cat, groups[cat]])
    }
    // Any remaining
    for (const [cat, items] of Object.entries(groups)) {
      if (!CATEGORY_ORDER.includes(cat)) sorted.push([cat, items])
    }
    return sorted
  }, [components, search])

  const toggle = (cat) => setExpanded((prev) => ({ ...prev, [cat]: !prev[cat] }))

  const handleDragStart = (e, comp) => {
    e.dataTransfer.setData('application/aria-component', JSON.stringify(comp))
    e.dataTransfer.effectAllowed = 'move'
    if (onDragStart) onDragStart(comp)
  }

  const handleMouseEnter = (e, comp) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const constraint = constraints[comp.id]
    setTooltip({
      comp,
      constraint,
      x: rect.right + 10,
      y: rect.top,
    })
  }

  const handleMouseLeave = () => setTooltip(null)

  return (
    <aside className="panel left">
      <h2>Components</h2>
      <input
        type="text"
        className="palette-search"
        placeholder="Search components (e.g. Hugging Face)..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />
      <div className="palette">
        {grouped.map(([cat, items]) => {
          // Auto-expand if searching, otherwise use state
          const isExpanded = search ? true : !!expanded[cat];
          const label = CATEGORY_LABELS[cat] || cat.replace(/_/g, ' ');
          
          return (
            <div key={cat} className="palette-group">
              <button className="palette-cat-header" onClick={() => toggle(cat)}>
                <span>{isExpanded ? '\u25BE' : '\u25B8'} {label}</span>
                <small>{items.length}</small>
              </button>
              {isExpanded && items.map((comp) => {
                const constraint = constraints[comp.id]
                const isSuggested = constraint?.suggested
                const isIncompatible = constraint?.compatible === false
                
                return (
                  <div
                    key={comp.id}
                    className={`palette-item cat-${comp.category || 'default'} ${isSuggested ? 'suggested' : ''} ${isIncompatible ? 'incompatible' : ''}`}
                    draggable
                    onDragStart={(e) => handleDragStart(e, comp)}
                    onMouseEnter={(e) => handleMouseEnter(e, comp)}
                    onMouseLeave={handleMouseLeave}
                  >
                    <span className="palette-item-name">
                      {isSuggested && <span className="suggested-star" title="Logical next step">★ </span>}
                      {comp.name}
                    </span>
                    <span className="palette-item-meta">
                      <span className="drag-hint" title="Drag to canvas">Drag</span>
                      {comp.performance?.has_params && <span className="tag param-tag">P</span>}
                    </span>
                  </div>
                )
              })}
            </div>
          );
        })}
        {Object.keys(grouped).length === 0 && (
          <p className="muted">No components match "{search}"</p>
        )}
      </div>

      {tooltip && (
        <div 
          className="palette-tooltip" 
          style={{ top: tooltip.y, left: tooltip.x }}
        >
          <div className="tooltip-header">
            <strong>{tooltip.comp.name}</strong>
            <span className="tooltip-cat">{tooltip.comp.category}</span>
          </div>
          <div className="tooltip-desc">
            {tooltip.comp.description || 'No description available.'}
          </div>
          {tooltip.constraint?.compatible === false && (
            <div className="tooltip-constraint-error">
              <strong>Incompatible:</strong>
              <ul>
                {tooltip.constraint.reasons.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            </div>
          )}
          {tooltip.constraint?.suggested && (
            <div className="tooltip-suggested-msg">
              Aria: This is a logical next step for your current architecture.
            </div>
          )}
          {tooltip.comp.inputs?.length > 0 && (
            <div className="tooltip-ports">
              <em>Inputs:</em> {tooltip.comp.inputs.map(i => i.name).join(', ')}
            </div>
          )}
        </div>
      )}
    </aside>
  )
}

export default memo(Palette)
