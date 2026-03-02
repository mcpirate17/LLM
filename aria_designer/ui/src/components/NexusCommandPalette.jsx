import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Search, Command, Cpu, Play, Save, Layout, Zap, X } from 'lucide-react';

export function NexusCommandPalette({ open, onClose, components, onAction }) {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef(null);

  const actions = useMemo(() => {
    const list = [
      { id: 'nav-dashboard', title: 'Switch to Research Dashboard', icon: Layout, category: 'Navigation', shortcut: 'G D' },
      { id: 'action-validate', title: 'Validate Graph', icon: Zap, category: 'Execution', shortcut: 'V' },
      { id: 'action-run', title: 'Run Preview', icon: Play, category: 'Execution', shortcut: 'R' },
      { id: 'action-save', title: 'Save Workflow', icon: Save, category: 'I/O', shortcut: 'Ctrl+S' },
    ];

    // Add components to the search list
    components.forEach(c => {
      list.push({
        id: `add-node-${c.id}`,
        title: `Add ${c.name} Node`,
        subtitle: c.category,
        icon: Cpu,
        category: 'Components',
        payload: c
      });
    });

    return list.filter(item => 
      item.title.toLowerCase().includes(query.toLowerCase()) || 
      item.category.toLowerCase().includes(query.toLowerCase())
    );
  }, [query, components]);

  useEffect(() => {
    if (open) {
      setQuery('');
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 10);
    }
  }, [open]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!open) return;
      
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex(prev => (prev + 1) % actions.length);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex(prev => (prev - 1 + actions.length) % actions.length);
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (actions[selectedIndex]) {
          onAction(actions[selectedIndex]);
          onClose();
        }
      } else if (e.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [open, actions, selectedIndex, onAction, onClose]);

  if (!open) return null;

  return (
    <div className="nexus-overlay" onClick={onClose}>
      <div className="nexus-palette" onClick={e => e.stopPropagation()}>
        <div className="nexus-search-box">
          <Search size={18} className="nexus-search-icon" />
          <input
            ref={inputRef}
            type="text"
            placeholder="Search components, actions, or navigation..."
            value={query}
            onChange={e => {
              setQuery(e.target.value);
              setSelectedIndex(0);
            }}
          />
          <div className="nexus-esc">ESC</div>
        </div>

        <div className="nexus-results">
          {actions.length === 0 ? (
            <div className="nexus-no-results">No matches found for "{query}"</div>
          ) : (
            actions.map((action, idx) => (
              <div
                key={action.id}
                className={`nexus-item ${idx === selectedIndex ? 'selected' : ''}`}
                onMouseEnter={() => setSelectedIndex(idx)}
                onClick={() => {
                  onAction(action);
                  onClose();
                }}
              >
                <action.icon size={16} className="nexus-item-icon" />
                <div className="nexus-item-info">
                  <div className="nexus-item-title">{action.title}</div>
                  {action.subtitle && <div className="nexus-item-subtitle">{action.subtitle}</div>}
                </div>
                {action.shortcut && <div className="nexus-item-shortcut">{action.shortcut}</div>}
                {action.category && <div className="nexus-item-category">{action.category}</div>}
              </div>
            ))
          )}
        </div>

        <div className="nexus-footer">
          <span><Command size={10} /> + K to toggle</span>
          <span>&uarr;&darr; to navigate</span>
          <span>&crarr; to select</span>
        </div>
      </div>
    </div>
  );
}

export default NexusCommandPalette;
