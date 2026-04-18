export const NAV_CATEGORIES = {
  workbench: {
    label: 'Workbench',
    tabs: ['command', 'experiments', 'discoveries', 'comparison'],
  },
  knowledge: {
    label: 'Knowledge',
    tabs: ['reports', 'trends', 'decisions', 'log'],
  },
  diagnostics: {
    label: 'Diagnostics',
    tabs: ['templates', 'components', 'infrastructure', 'perf', 'references'],
  },
};

export const TAB_LABELS = {
  command: 'Command',
  trends: 'Analytics',
  experiments: 'Experiments',
  discoveries: 'Discoveries',
  comparison: 'Comparison',
  templates: 'Template & Slots',
  infrastructure: 'Infrastructure',
  components: 'Components',
  perf: 'Optimization',
  reports: 'Reports',
  references: 'References',
  decisions: 'Decisions',
  log: 'Log',
};

export const TAB_TIPS = {
  command: 'Control center — start/stop experiments, see live status (1)',
  trends: 'Analytics: trends, learning signals, and diagnostic charts (2)',
  experiments: 'Browse all experiments and their results (3)',
  discoveries: 'Best architectures found so far, ranked by tier (4)',
  comparison: 'Side-by-side architecture comparison (5)',
  templates: 'Dedicated page for template success, weak slots, fast-lane fairness, and structural trends',
  infrastructure: 'Pipeline health, alerts, live stream, throughput, resources',
  components: 'Component health, op analytics, grammar evolution, insights',
  perf: 'System performance and optimization metrics (6)',
  reports: 'Publishable findings, campaigns, and knowledge base (7)',
  references: 'Reference models (GPT-2, Mamba, etc.) baselines (8)',
  decisions: 'Recent automated research decision traces (9)',
  log: 'Raw notebook entries and cycle timeline (0)',
};

