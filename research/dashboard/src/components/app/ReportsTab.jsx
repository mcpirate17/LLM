import React, { Suspense } from 'react';
import { LazyFallback } from './AppShellShared';
import { CampaignView, KnowledgeBase, ResearchReport } from './lazyComponents';
import { DeferredInsightsSection } from './AppShellSections';

function renderReportsLoading() {
  return (
    <div className="card" style={{ marginTop: 16 }}>
      <p style={{ color: 'var(--text-muted)', margin: 0 }}>
        Loading campaigns and knowledge base...
      </p>
    </div>
  );
}

export default function ReportsTab({
  eligibilityByResultId,
  handleHypothesisHandoff,
  handleInvestigate,
  handleQueueAdd,
  handleQueueRemove,
  handleSelectCampaign,
  handleSelectExperiment,
  handleSelectProgram,
  handleValidate,
  onOpenDesignerForResult,
  queuedResultIds,
  reportsCampaignsVisible,
  reportsDeferredReady,
  reportsKnowledgeVisible,
  selectedCampaignId,
  setReportsCampaignsVisible,
  setReportsKnowledgeVisible,
}) {
  return (
    <Suspense fallback={<LazyFallback />}>
      <ResearchReport
        onSelectProgram={handleSelectProgram}
        onSelectExperiment={handleSelectExperiment}
        onInvestigate={handleInvestigate}
        onValidate={handleValidate}
        onOpenInDesigner={onOpenDesignerForResult}
        onQueueAdd={handleQueueAdd}
        onQueueRemove={handleQueueRemove}
        queuedResultIds={queuedResultIds}
        eligibilityByResultId={eligibilityByResultId}
        onHypothesisHandoff={handleHypothesisHandoff}
      />
      {reportsDeferredReady ? (
        <>
          <DeferredInsightsSection
            title="Campaigns"
            visible={reportsCampaignsVisible}
            onLoad={() => setReportsCampaignsVisible(true)}
            emptyText="Campaign details are available on demand to keep the reports page responsive."
          >
            <CampaignView
              onSelectExperiment={handleSelectExperiment}
              selectedCampaignId={selectedCampaignId}
              onCampaignIdClear={() => handleSelectCampaign(null)}
              onHypothesisHandoff={handleHypothesisHandoff}
            />
          </DeferredInsightsSection>
          <DeferredInsightsSection
            title="Knowledge Base"
            visible={reportsKnowledgeVisible}
            onLoad={() => setReportsKnowledgeVisible(true)}
            emptyText="Knowledge clustering is deferred until requested so the reports landing page stays light."
          >
            <KnowledgeBase onSelectExperiment={handleSelectExperiment} />
          </DeferredInsightsSection>
        </>
      ) : renderReportsLoading()}
    </Suspense>
  );
}
