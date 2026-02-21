import React, { useState } from 'react';
import ReportGallery from './ReportGallery';
import ReportDetail from './ReportDetail';

function ResearchReport(props) {
  const [selectedScope, setSelectedScope] = useState(null);

  if (selectedScope) {
    return (
      <ReportDetail
        key={selectedScope.id}
        scope={selectedScope}
        onBack={() => setSelectedScope(null)}
        {...props}
      />
    );
  }

  return <ReportGallery onSelectScope={setSelectedScope} />;
}

export default ResearchReport;
