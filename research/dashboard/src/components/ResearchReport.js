import React, { useCallback, useState } from 'react';
import ReportGallery from './ReportGallery';
import ReportDetail from './ReportDetail';

function ResearchReport(props) {
  const [selectedScope, setSelectedScope] = useState(null);
  const handleSelectScope = useCallback((scope) => {
    setSelectedScope(scope);
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <ReportGallery
        selectedScopeId={selectedScope?.id || null}
        onSelectScope={handleSelectScope}
      />
      {selectedScope && (
        <ReportDetail
          key={selectedScope.id}
          scope={selectedScope}
          onBack={() => setSelectedScope(null)}
          {...props}
        />
      )}
    </div>
  );
}

export default ResearchReport;
