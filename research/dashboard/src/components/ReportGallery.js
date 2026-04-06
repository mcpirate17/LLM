import React from 'react';
import { useReportGallery } from '../hooks/useReportGallery';
import ReportCard from './report/ReportCard';

export default function ReportGallery({ onSelectScope, selectedScopeId = null }) {
  const { cards, loading } = useReportGallery();

  if (loading) {
    return <div className="card"><p style={{ color: 'var(--text-muted)' }}>Loading report gallery...</p></div>;
  }

  const timeCards = cards.filter(c => c.section === 'time');
  const themeCards = cards.filter(c => c.section === 'theme');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="card">
        <div className="card-title" style={{ marginBottom: 4 }}>Research Reports</div>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          Select a time period or research theme to view a scoped report.
        </div>
      </div>

      {/* Time-based reports */}
      <div>
        <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 10 }}>
          By Time Period
        </div>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
          gap: 16,
        }}>
          {timeCards.map(card => (
            <ReportCard
              key={card.id}
              label={card.label}
              stats={card.stats}
              highlight={card.highlight}
              selected={selectedScopeId === card.scope?.id}
              onClick={() => onSelectScope(card.scope)}
            />
          ))}
        </div>
      </div>

      {/* Theme-based reports */}
      <div>
        <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 10 }}>
          By Research Theme
        </div>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
          gap: 16,
        }}>
          {themeCards.map(card => (
            <ReportCard
              key={card.id}
              label={card.label}
              stats={card.stats}
              selected={selectedScopeId === card.scope?.id}
              onClick={() => onSelectScope(card.scope)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
