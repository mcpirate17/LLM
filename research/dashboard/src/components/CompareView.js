import React, { useState, useEffect, useMemo } from 'react';
import { apiCall } from "../services/apiService";
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import ChartActions from './ChartActions';

function CompareView({ comparisonList, onRemoveProgram, onSelectProgram }) {
  const [details, setDetails] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    async function fetchDetails() {
      if (comparisonList.length === 0) {
        setDetails([]);
        return;
      }
      setLoading(true);
      try {
        const results = await Promise.all(
          comparisonList.map(id => apiCall(`/api/reproducibility-manifest/${id}`).then(r => r.json()))
        );
        setDetails(results.filter(r => !r.error));
      } catch (e) {
        console.error("Failed to fetch comparison details", e);
      }
      setLoading(false);
    }
    fetchDetails();
  }, [comparisonList]);

  if (comparisonList.length === 0) {
    return (
      <div className="card" style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
        <h3>No architectures selected for comparison</h3>
        <p>Add architectures to compare from the Program Detail or Leaderboard views.</p>
      </div>
    );
  }

  // Transform for Radar Chart
  const metrics = [
    { key: 'loss_ratio', label: 'Loss Ratio', invert: true },
    { key: 'novelty_score', label: 'Novelty', invert: false },
    { key: 'baseline_loss_ratio', label: 'vs Baseline', invert: true },
    { key: 'param_count', label: 'Efficiency', invert: true },
  ];

  const radarData = metrics.map(m => {
    const row = { subject: m.label };
    details.forEach(d => {
      let val = d.outcomes?.[m.key] || d.program?.[m.key] || 0;
      if (m.key === 'param_count') {
          // Normalise params: 1M -> 1.0, 100M -> 0.0
          val = Math.max(0, Math.min(1, (8 - Math.log10(val || 1)) / 2));
      } else if (m.invert) {
          val = Math.max(0, 1 - val);
      }
      row[d.result_id?.slice(0, 8) || 'Unknown'] = val;
    });
    return row;
  });

  const colors = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#bc8cff'];

  // Contextual comparison actions
  const comparisonActions = useMemo(() => {
    if (details.length < 2) return [];
    const result = [];

    // Check if one model dominates another on most metrics
    for (let i = 0; i < details.length; i++) {
      for (let j = i + 1; j < details.length; j++) {
        const a = details[i], b = details[j];
        let aWins = 0, bWins = 0;
        const metricKeys = ['loss_ratio', 'novelty_score', 'baseline_loss_ratio'];
        for (const m of metricKeys) {
          const va = a.outcomes?.[m] ?? a.program?.[m] ?? null;
          const vb = b.outcomes?.[m] ?? b.program?.[m] ?? null;
          if (va == null || vb == null) continue;
          const lower = m !== 'novelty_score'; // lower is better for loss metrics
          if (lower ? va < vb : va > vb) aWins++;
          else if (lower ? vb < va : vb > va) bWins++;
        }
        if (aWins >= 3 && bWins === 0) {
          result.push({
            id: `dom-${i}-${j}`,
            label: `${a.result_id?.slice(0, 8)} dominates ${b.result_id?.slice(0, 8)} on ${aWins}/${metricKeys.length} metrics`,
            detail: `Consider dropping ${b.result_id?.slice(0, 8)}`,
            color: colors[i % colors.length],
            onClick: () => onSelectProgram?.(a.result_id),
          });
        } else if (bWins >= 3 && aWins === 0) {
          result.push({
            id: `dom-${j}-${i}`,
            label: `${b.result_id?.slice(0, 8)} dominates ${a.result_id?.slice(0, 8)} on ${bWins}/${metricKeys.length} metrics`,
            detail: `Consider dropping ${a.result_id?.slice(0, 8)}`,
            color: colors[j % colors.length],
            onClick: () => onSelectProgram?.(b.result_id),
          });
        }
      }
    }

    // Highest novelty but worst loss
    if (details.length >= 2) {
      const byNovelty = [...details].sort((a, b) => (b.outcomes?.novelty_score || 0) - (a.outcomes?.novelty_score || 0));
      const byLoss = [...details].sort((a, b) => (a.outcomes?.loss_ratio || 1) - (b.outcomes?.loss_ratio || 1));
      const highestNovelty = byNovelty[0];
      const bestLoss = byLoss[0];
      if (highestNovelty.result_id !== bestLoss.result_id && (highestNovelty.outcomes?.novelty_score || 0) > 0) {
        result.push({
          id: 'novelty-vs-loss',
          label: `${highestNovelty.result_id?.slice(0, 8)} highest novelty but not best loss \u2014 investigate hybrid?`,
          color: 'var(--accent-purple)',
          onClick: () => onSelectProgram?.(highestNovelty.result_id),
        });
      }
    }

    return result.slice(0, 3);
  }, [details, onSelectProgram]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 3fr', gap: 16 }}>
      <div className="card" style={{ padding: 12 }}>
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>Selected ({comparisonList.length})</div>
        {details.map((d, i) => (
          <div key={d.result_id} style={{ 
            padding: '8px 10px', borderRadius: 6, background: 'var(--bg-secondary)', 
            borderLeft: `4px solid ${colors[i % colors.length]}`, marginBottom: 8,
            fontSize: 12, position: 'relative'
          }}>
            <div style={{ fontWeight: 600, marginBottom: 2 }}>{d.result_id?.slice(0, 12)}</div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{d.program?.architecture_family || 'Custom'}</div>
            <button 
              onClick={() => onRemoveProgram(d.result_id)}
              style={{ position: 'absolute', top: 4, right: 4, background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
            >&times;</button>
          </div>
        ))}
      </div>

      <div className="card" style={{ padding: 16 }}>
        <div style={{ height: 400 }}>
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
              <PolarGrid stroke="var(--border)" />
              <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--text-muted)', fontSize: 12 }} />
              <PolarRadiusAxis angle={30} domain={[0, 1]} tick={false} axisLine={false} />
              {details.map((d, i) => (
                <Radar
                  key={d.result_id}
                  name={d.result_id?.slice(0, 8)}
                  dataKey={d.result_id?.slice(0, 8)}
                  stroke={colors[i % colors.length]}
                  fill={colors[i % colors.length]}
                  fillOpacity={0.3}
                />
              ))}
              <Legend />
              <Tooltip 
                contentStyle={{ background: 'var(--bg-primary)', border: '1px solid var(--border)' }}
                itemStyle={{ fontSize: 12 }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
        <ChartActions actions={comparisonActions} />

        <div style={{ marginTop: 24, overflowX: 'auto' }}>
          <table className="data-table table-compact">
            <thead>
              <tr style={{ borderBottom: '2px solid var(--border)' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Metric</th>
                {details.map((d, i) => (
                  <th key={d.result_id} style={{ color: colors[i % colors.length], padding: 8 }}>
                    {d.result_id?.slice(0, 8)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ padding: 8, fontWeight: 600 }}>Validation LR</td>
                {details.map(d => <td key={d.result_id} style={{ padding: 8, textAlign: 'center' }}>{d.outcomes?.validation_loss_ratio?.toFixed(4) || '--'}</td>)}
              </tr>
              <tr>
                <td style={{ padding: 8, fontWeight: 600 }}>Discovery LR</td>
                {details.map(d => <td key={d.result_id} style={{ padding: 8, textAlign: 'center' }}>{d.outcomes?.discovery_loss_ratio?.toFixed(4) || '--'}</td>)}
              </tr>
              <tr>
                <td style={{ padding: 8, fontWeight: 600 }}>Novelty</td>
                {details.map(d => <td key={d.result_id} style={{ padding: 8, textAlign: 'center' }}>{d.outcomes?.novelty_score?.toFixed(3) || '--'}</td>)}
              </tr>
              <tr>
                <td style={{ padding: 8, fontWeight: 600 }}>Params</td>
                {details.map(d => <td key={d.result_id} style={{ padding: 8, textAlign: 'center' }}>{(d.program?.param_count / 1e6).toFixed(1)}M</td>)}
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default CompareView;
