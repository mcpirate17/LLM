import React, { useState } from 'react';

const SECTIONS = [
  {
    title: 'Getting Started',
    content: `**Start the dashboard:**
\`\`\`
python -m research --mode=dashboard
\`\`\`

**Quick start workflow:**
1. Open the dashboard in your browser (default: http://localhost:5000)
2. On the Overview tab, configure your experiment in the Control Panel
3. Click "Run Experiment" to start a single synthesis run
4. Watch results stream in via the Live Feed
5. After completion, explore results in the Experiments and Programs tabs

**Continuous research mode** runs experiments back-to-back with Aria auto-generating hypotheses between runs.`
  },
  {
    title: 'Experiment Modes',
    content: `**Single Experiment** - Generate N programs, evaluate through the pipeline, analyze results. Good for testing specific hypotheses.

**Continuous Research** - Run experiments back-to-back. Aria formulates new hypotheses between runs based on what she's learned. Set max experiments to control duration.

**Evolution Search** - Uses evolutionary algorithms (tournament selection, mutation, crossover) to search the architecture space. Optimizes a blend of fitness (learning ability) and novelty. Configure population size, generations, mutation/crossover rates, and elitism.

**Novelty Search** - Evolutionary search that specifically rewards behavioral novelty. Maintains an archive of seen behaviors and selects for architectures that behave differently from everything seen before. Key parameters: archive size, K-nearest neighbors, novelty threshold.`
  },
  {
    title: 'Evaluation Pipeline',
    content: `Programs pass through a multi-stage evaluation funnel:

**Stage 0 (Validation + Compilation)** - Graph structure validated, compiled into a PyTorch model. Tests: valid topology, gradient paths exist, parameter count reasonable.

**Stage 0.5 (Sandbox)** - Forward/backward pass stability. Tests: no NaN/Inf outputs, stable gradients, reasonable output range, handles extreme inputs.

**Stage 1 (Micro-training)** - 500 steps of next-token prediction on random data. A program "passes" if its loss ratio (final/initial loss) drops below 0.8, proving it can learn.

**Novelty Scoring** - Behavioral fingerprinting captures how the architecture processes information (interaction patterns, representation geometry, CKA similarity to known architectures). Novelty is distance from known patterns.

**Baseline Comparison** - Stage 1 survivors are compared against a standard transformer trained under identical conditions.`
  },
  {
    title: 'Metrics Guide',
    content: `**Loss Ratio** - final_loss / initial_loss. Below 0.8 = learning. Lower is better.

**Novelty Score** - 0-1 composite of structural and behavioral novelty. Above 0.5 = genuinely novel.

**CKA Similarity** - Centered Kernel Alignment vs transformer/SSM/conv baselines. Low CKA = behaves very differently from known architectures.

**Fingerprint Dimensions:**
- Interaction locality/sparsity/symmetry/hierarchy - How the architecture routes information
- Isotropy/rank ratio - Representation geometry
- Jacobian spectral norm/effective rank - Sensitivity properties
- Intrinsic dimensionality - Effective capacity

**FLOPs** - Forward pass floating-point operations. Used for efficiency frontier analysis.

**Baseline Ratio** - Loss compared to transformer baseline. Below 1.0 = beats the baseline.`
  },
  {
    title: 'Grammar Learning',
    content: `The system learns which operations and structures tend to produce successful architectures.

**Op Success Rates** - Each primitive operation is tracked for S0/S0.5/S1 pass rates. Operations that consistently produce learnable architectures get higher success rates.

**Grammar Weights** - Category weights control how often each type of operation is sampled during graph generation. Weights are learned from historical success:
\`\`\`
weight = 1.0 + s1_rate * 3.0 + avg_novelty * 1.0
\`\`\`
Clamped to [0.3, 5.0]. Higher weight = sampled more often.

**Learning Log** - Audit trail of weight changes applied before each experiment. Viewable in the Learning tab.`
  },
  {
    title: 'LLM Configuration',
    content: `Aria uses an LLM backend for analysis, hypothesis generation, and summaries. Without one, she falls back to rule-based methods.

**Anthropic (Claude):**
\`\`\`
export ARIA_LLM_BACKEND=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export ANTHROPIC_MODEL=claude-sonnet-4-5-20250929  # optional
\`\`\`

**OpenAI:**
\`\`\`
export ARIA_LLM_BACKEND=openai
export OPENAI_API_KEY=sk-...
\`\`\`

**Ollama (local):**
\`\`\`
export ARIA_LLM_BACKEND=ollama
export OLLAMA_API_URL=http://localhost:11434  # default
\`\`\`

The system status badge on the dashboard shows whether an LLM backend is active.`
  },
  {
    title: 'Dashboard Tabs',
    content: `**Overview** - Aria's status, experiment control panel, summary stats, live feed during runs, top programs and insights.

**Experiments** - List of all experiments with status, pass rates, and key metrics. Click to drill down.

**Programs** - Top programs ranked by novelty or loss ratio. Click for full detail including behavioral fingerprint and graph structure.

**Trends** - Cross-experiment charts showing S1 pass rate, novelty, and loss ratio over time.

**Learning** - Grammar weight evolution, op success rates, learning log audit trail.

**Notebook** - Raw lab notebook entries: hypotheses, observations, analyses, errors, and insights.

**Insights** - Data-driven insights from analytics: top/bottom ops, structural correlations, failure patterns, winning combinations.

**Help** - This panel.`
  },
];

function HelpPanel() {
  const [openSection, setOpenSection] = useState(null);

  const toggleSection = (idx) => {
    setOpenSection(openSection === idx ? null : idx);
  };

  return (
    <div className="card help-panel">
      <div className="card-title">Help & Documentation</div>
      <p className="help-intro">
        Welcome to Aria's Lab. Click a section below to learn more.
      </p>
      {SECTIONS.map((section, idx) => (
        <div key={idx} className="help-section">
          <button
            className={`help-section-header ${openSection === idx ? 'open' : ''}`}
            onClick={() => toggleSection(idx)}
          >
            <span>{openSection === idx ? '\u25BC' : '\u25B6'} {section.title}</span>
          </button>
          {openSection === idx && (
            <div className="help-section-content">
              <SimpleMarkdown text={section.content} />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function SimpleMarkdown({ text }) {
  const lines = text.split('\n');
  const elements = [];
  let inCodeBlock = false;
  let codeLines = [];
  let key = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (line.startsWith('```')) {
      if (inCodeBlock) {
        elements.push(
          <pre key={key++} className="help-code">
            <code>{codeLines.join('\n')}</code>
          </pre>
        );
        codeLines = [];
        inCodeBlock = false;
      } else {
        inCodeBlock = true;
      }
      continue;
    }

    if (inCodeBlock) {
      codeLines.push(line);
      continue;
    }

    if (line.trim() === '') {
      elements.push(<br key={key++} />);
      continue;
    }

    elements.push(<p key={key++} className="help-line">{formatInline(line)}</p>);
  }

  return <div>{elements}</div>;
}

function formatInline(text) {
  const parts = [];
  let remaining = text;
  let key = 0;

  while (remaining.length > 0) {
    // Bold
    const boldMatch = remaining.match(/\*\*(.+?)\*\*/);
    // Inline code
    const codeMatch = remaining.match(/`([^`]+)`/);

    let firstMatch = null;
    let firstIdx = remaining.length;

    if (boldMatch && boldMatch.index < firstIdx) {
      firstMatch = { type: 'bold', match: boldMatch };
      firstIdx = boldMatch.index;
    }
    if (codeMatch && codeMatch.index < firstIdx) {
      firstMatch = { type: 'code', match: codeMatch };
      firstIdx = codeMatch.index;
    }

    if (!firstMatch) {
      parts.push(remaining);
      break;
    }

    if (firstIdx > 0) {
      parts.push(remaining.slice(0, firstIdx));
    }

    const m = firstMatch.match;
    if (firstMatch.type === 'bold') {
      parts.push(<strong key={key++}>{m[1]}</strong>);
    } else {
      parts.push(<code key={key++} className="help-inline-code">{m[1]}</code>);
    }

    remaining = remaining.slice(firstIdx + m[0].length);
  }

  return parts;
}

export default HelpPanel;
