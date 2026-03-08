import React from 'react';
import Palette from './Palette';
import Inspector from './InspectorMain';
import RunResultsPanel from './RunResultsPanel';
import PatchPanel from './PatchPanel';
import '../styles/Sidebar.css';

const Sidebar = ({ 
  rightPanelTab, 
  setRightPanelTab, 
  components, 
  selectedNode, 
  side = 'right',
  style,
  nodeCount,
  edgeCount,
  evalState,
  proposals,
  onApplyPatch,
  onRejectPatch,
  onPreviewPatch,
  previewPatchId
}) => {
  return (
    <aside className={`app-sidebar ${side === 'left' ? 'left' : 'right'}`} style={style}>
      <div className="sidebar-tabs">
        <button 
          className={rightPanelTab === 'palette' ? 'active' : ''} 
          onClick={() => setRightPanelTab('palette')}
        >
          Palette
        </button>
        <button 
          className={rightPanelTab === 'inspector' ? 'active' : ''} 
          onClick={() => setRightPanelTab('inspector')}
        >
          Inspector
        </button>
        <button 
          className={rightPanelTab === 'results' ? 'active' : ''} 
          onClick={() => setRightPanelTab('results')}
        >
          Results
        </button>
        <button 
          className={rightPanelTab === 'proposals' ? 'active' : ''} 
          onClick={() => setRightPanelTab('proposals')}
        >
          Proposals {proposals.length > 0 ? `(${proposals.length})` : ''}
        </button>
      </div>

      <div className="sidebar-content">
        {rightPanelTab === 'palette' && (
          <Palette components={components} />
        )}
        {rightPanelTab === 'inspector' && (
          <div className="panel right" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Inspector 
              selectedNode={selectedNode} 
              nodeCount={nodeCount}
              edgeCount={edgeCount}
            />
          </div>
        )}
        {rightPanelTab === 'results' && (
          <RunResultsPanel evalState={evalState} />
        )}
        {rightPanelTab === 'proposals' && (
          <PatchPanel 
            proposals={proposals} 
            onApply={onApplyPatch} 
            onReject={onRejectPatch}
            onPreview={onPreviewPatch}
            previewPatchId={previewPatchId}
          />
        )}
      </div>
    </aside>
  );
};

export default Sidebar;
