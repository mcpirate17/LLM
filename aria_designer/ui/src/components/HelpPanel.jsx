import { memo, useCallback, useEffect, useState } from 'react';
import { X, Search, BookOpen, Lightbulb, Layout } from 'lucide-react';
import '../styles/HelpPanel.css';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8091';

const GETTING_STARTED = [
  { title: 'Drop components', desc: 'Drag components from the left palette onto the canvas.' },
  { title: 'Connect nodes', desc: 'Drag from an output port to an input port to create edges.' },
  { title: 'Configure params', desc: 'Select a node and edit parameters in the right inspector panel.' },
  { title: 'Validate', desc: 'Click "Step 1: Validate" to check your graph for errors.' },
  { title: 'Compile & Test', desc: 'Steps 2 and 3 compile to a PyTorch module and run a forward pass.' },
  { title: 'Ask Aria', desc: 'Click "Ask Aria" for AI-powered suggestions and automatic patching.' },
  { title: 'Save & Discover', desc: 'Save your workflow. The fingerprint links to the research leaderboard.' },
];

function HelpPanel({ isOpen, onClose }) {
  const [activeTab, setActiveTab] = useState('start');
  const [searchQuery, setSearchQuery] = useState('');
  const [componentTips, setComponentTips] = useState(null);
  const [patterns, setPatterns] = useState(null);
  const [loadingTips, setLoadingTips] = useState(false);

  useEffect(() => {
    if (!isOpen || patterns) return;
    const ac = new AbortController();
    fetch(`${API_BASE}/api/v1/help/patterns`, { signal: ac.signal })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => { if (data) setPatterns(data); })
      .catch(() => {});
    return () => ac.abort();
  }, [isOpen, patterns]);

  const searchComponent = useCallback((query) => {
    setSearchQuery(query);
    if (!query || query.length < 2) {
      setComponentTips(null);
      return;
    }
    setLoadingTips(true);
    fetch(`${API_BASE}/api/v1/help/component/${encodeURIComponent(query)}/tips`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => { setComponentTips(data); setLoadingTips(false); })
      .catch(() => { setLoadingTips(false); });
  }, []);

  if (!isOpen) return null;

  return (
    <div className="help-panel-overlay" onClick={onClose}>
      <div className="help-panel" onClick={(e) => e.stopPropagation()}>
        <div className="help-panel-header">
          <h2>Help</h2>
          <button className="help-close-btn" onClick={onClose}><X size={18} /></button>
        </div>

        <div className="help-tabs">
          <button
            className={`help-tab ${activeTab === 'start' ? 'active' : ''}`}
            onClick={() => setActiveTab('start')}
          >
            <BookOpen size={14} /> Getting Started
          </button>
          <button
            className={`help-tab ${activeTab === 'guide' ? 'active' : ''}`}
            onClick={() => setActiveTab('guide')}
          >
            <Search size={14} /> Component Guide
          </button>
          <button
            className={`help-tab ${activeTab === 'patterns' ? 'active' : ''}`}
            onClick={() => setActiveTab('patterns')}
          >
            <Layout size={14} /> Patterns
          </button>
        </div>

        <div className="help-content">
          {activeTab === 'start' && (
            <div className="help-start">
              {GETTING_STARTED.map((item, i) => (
                <div key={i} className="help-step">
                  <span className="help-step-num">{i + 1}</span>
                  <div>
                    <div className="help-step-title">{item.title}</div>
                    <div className="help-step-desc">{item.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab === 'guide' && (
            <div className="help-guide">
              <div className="help-search-bar">
                <Search size={14} />
                <input
                  type="text"
                  placeholder="Search component (e.g. rmsnorm, attention)..."
                  value={searchQuery}
                  onChange={(e) => searchComponent(e.target.value)}
                  autoFocus
                />
              </div>
              {loadingTips && <div className="help-loading">Loading...</div>}
              {componentTips && (
                <div className="help-tips-result">
                  <h3>{componentTips.component_id}</h3>
                  {componentTips.categories?.length > 0 && (
                    <div className="help-categories">
                      {componentTips.categories.map((c) => (
                        <span key={c} className="tip-chip tip-chip-cat">{c}</span>
                      ))}
                    </div>
                  )}
                  {componentTips.works_well_with?.length > 0 && (
                    <div className="help-compat-section">
                      <h4>Works well with</h4>
                      <div className="tip-chips">
                        {componentTips.works_well_with.map((id) => (
                          <span
                            key={id}
                            className="tip-chip tip-chip-good clickable"
                            onClick={() => searchComponent(id)}
                          >
                            {id}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {componentTips.avoid_with?.length > 0 && (
                    <div className="help-compat-section">
                      <h4>Avoid combining with</h4>
                      <div className="tip-chips">
                        {componentTips.avoid_with.map((id) => (
                          <span key={id} className="tip-chip tip-chip-bad">{id}</span>
                        ))}
                      </div>
                    </div>
                  )}
                  {componentTips.patterns?.length > 0 && (
                    <div className="help-compat-section">
                      <h4>Usage tips</h4>
                      <ul className="help-tip-list">
                        {componentTips.patterns.map((p, i) => (
                          <li key={i}>{p}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {componentTips.leaderboard_usage && (
                    <div className="tip-badge tip-badge-leaderboard">
                      <Lightbulb size={12} /> {componentTips.leaderboard_usage}
                    </div>
                  )}
                </div>
              )}
              {!componentTips && !loadingTips && searchQuery.length >= 2 && (
                <div className="help-no-results">No tips found for "{searchQuery}"</div>
              )}
              {!searchQuery && (
                <div className="help-guide-placeholder">
                  Type a component name to see compatibility info, usage tips, and research data.
                </div>
              )}
            </div>
          )}

          {activeTab === 'patterns' && (
            <div className="help-patterns">
              {patterns?.patterns?.map((p, i) => (
                <div key={i} className="help-pattern-card">
                  <h3>{p.name}</h3>
                  <p>{p.description}</p>
                  <div className="tip-chips">
                    {p.components.map((c) => (
                      <span key={c} className="tip-chip tip-chip-good">{c}</span>
                    ))}
                  </div>
                </div>
              ))}
              {patterns?.tips?.length > 0 && (
                <div className="help-general-tips">
                  <h3>General Tips</h3>
                  <ul className="help-tip-list">
                    {patterns.tips.map((t, i) => (
                      <li key={i}>{t}</li>
                    ))}
                  </ul>
                </div>
              )}
              {!patterns && <div className="help-loading">Loading patterns...</div>}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default memo(HelpPanel);
