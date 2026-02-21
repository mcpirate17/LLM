import { useState, useEffect, useMemo } from 'react';
import { useAriaData } from './useAriaData';

const API_BASE = process.env.REACT_APP_API_URL || '';

function startOfDay(date) {
  const d = new Date(date);
  d.setHours(0, 0, 0, 0);
  return d;
}

function getMondayOfWeek(date) {
  const d = new Date(date);
  const day = d.getDay();
  const diff = d.getDate() - day + (day === 0 ? -6 : 1);
  d.setDate(diff);
  d.setHours(0, 0, 0, 0);
  return d;
}

function filterByDateRange(experiments, startDate) {
  const cutoff = startDate.getTime();
  return experiments.filter(exp => {
    const ts = exp.created_at ? new Date(exp.created_at).getTime() : 0;
    return ts >= cutoff;
  });
}

function computeStats(experiments) {
  let totalPrograms = 0;
  let s1Survivors = 0;

  for (const exp of experiments) {
    totalPrograms += exp.n_programs_generated || 0;
    s1Survivors += exp.n_stage1_passed || 0;
  }

  const passRate = totalPrograms > 0 ? s1Survivors / totalPrograms : 0;

  return {
    experiments: experiments.length,
    programs: totalPrograms,
    s1Survivors,
    passRate,
  };
}

export function useReportGallery() {
  const [trends, setTrends] = useState([]);
  const [loading, setLoading] = useState(true);
  const { summary } = useAriaData() || {};

  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/api/trends/context`)
      .then(r => r.ok ? r.json() : null)
      .then(payload => {
        if (!active) return;
        setTrends(Array.isArray(payload?.trends) ? payload.trends : []);
        setLoading(false);
      })
      .catch(() => {
        if (active) setLoading(false);
      });
    return () => { active = false; };
  }, []);

  const cards = useMemo(() => {
    const now = new Date();

    // Time-based cards
    const yesterday = startOfDay(new Date(now.getTime() - 24 * 60 * 60 * 1000));
    const monday = getMondayOfWeek(now);

    const last24h = filterByDateRange(trends, yesterday);
    const thisWeek = filterByDateRange(trends, monday);

    const last24hStats = computeStats(last24h);
    const thisWeekStats = computeStats(thisWeek);

    // All-Time uses summary from useAriaData for consistency
    const allTimeStats = summary ? {
      experiments: summary.total_experiments || 0,
      programs: summary.total_programs_evaluated || 0,
      s1Survivors: summary.stage1_survivors ?? summary.total_s1_passed ?? 0,
      passRate: (summary.total_programs_evaluated || 0) > 0
        ? (summary.stage1_survivors ?? summary.total_s1_passed ?? 0) / (summary.total_programs_evaluated || 0)
        : 0,
    } : computeStats(trends);

    const timeCards = [
      {
        id: 'last-24h',
        label: 'Last 24 Hours',
        section: 'time',
        stats: last24hStats,
        scope: {
          id: 'last-24h',
          label: 'Last 24 Hours',
          params: { start_date: yesterday.toISOString().slice(0, 10) },
        },
      },
      {
        id: 'this-week',
        label: 'This Week',
        section: 'time',
        stats: thisWeekStats,
        scope: {
          id: 'this-week',
          label: 'This Week',
          params: { start_date: monday.toISOString().slice(0, 10) },
        },
      },
      {
        id: 'all-time',
        label: 'All Time',
        section: 'time',
        stats: allTimeStats,
        highlight: true,
        scope: {
          id: 'all-time',
          label: 'All Time Report',
          params: null, // null means use /api/report
        },
      },
    ];

    // Theme-based cards (experiment count only from trends)
    const themes = [
      { id: 'sparsity', label: 'Sparsity Research', theme: 'sparsity' },
      { id: 'compression', label: 'Compression', theme: 'compression' },
      { id: 'routing', label: 'Routing', theme: 'routing' },
      { id: 'mathspace', label: 'Mathspace', theme: 'mathspace' },
    ];

    const themeCards = themes.map(t => {
      // We can't reliably filter trends by theme client-side since trends data
      // doesn't have a theme field. Show total experiment count and let the
      // scoped report endpoint do the filtering.
      return {
        id: t.id,
        label: t.label,
        section: 'theme',
        stats: { experiments: null, programs: null, s1Survivors: null, passRate: null },
        scope: {
          id: t.id,
          label: `${t.label} Report`,
          params: { theme: t.theme },
        },
      };
    });

    return [...timeCards, ...themeCards];
  }, [trends, summary]);

  return { cards, loading };
}
