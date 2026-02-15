# AI Scientist — Autonomous Architecture Discovery

An autonomous system for discovering novel computation patterns through grammar-based program synthesis, multi-stage evaluation, and a learning feedback loop that improves its own search strategy over time.

## What It Does

The AI scientist (Dr. Aria Nexus) autonomously:

1. **Synthesizes** random computation graphs from ~50 tensor operation primitives spanning elementwise, reduction, linear algebra, frequency domain, and exotic mathematical spaces (tropical geometry, hyperbolic space, Clifford algebras, p-adic numbers)
2. **Evaluates** candidates through a multi-stage funnel that filters thousands of programs down to genuine novel architectures
3. **Learns** which operations, structures, and combinations correlate with success — grammar weights adapt based on experimental history
4. **Analyzes** results with behavioral fingerprinting, baseline comparison, and efficiency frontier analysis

## Evaluation Funnel

Each synthesized program passes through progressive filters:

- **Stage 0** — Compilation + forward/backward pass (~seconds). Captures compile time, memory, gradient health, output pathology flags.
- **Stage 0.5** — Numerical stability probe. Tests multiple input distributions, measures stability score.
- **Stage 1** — Micro-training: 500 steps of AdamW on random data. Must show >20% loss reduction. Captures full training curve, gradient norm statistics, throughput.
- **Novelty scoring** — Behavioral fingerprinting (interaction patterns, representation geometry, Jacobian spectrum, CKA similarity vs transformer/SSM/conv). Programs must be behaviorally distinct from known architectures.
- **Baseline comparison** — Loss compared to a cached vanilla 2-layer transformer trained under identical conditions.
- **FLOP estimation** — Forward-pass FLOPs estimated per operation type for efficiency analysis.

~70+ metrics are captured per program and stored in SQLite for analysis.

## How It Learns

After each experiment, the analytics engine:

- Computes **per-operation success rates** (which ops appear in Stage 1 survivors?)
- Finds **structural correlations** (graph depth, op count, math spaces vs success)
- Identifies **winning op combinations** (co-occurring ops in survivors)
- Computes **learned grammar weights** from historical data — categories with higher S1 rates and novelty get higher synthesis probability
- Logs all weight changes to an **audit trail** for reproducibility
- Builds a **Pareto efficiency frontier** on loss vs FLOPs

Grammar weights start at defaults and evolve as experiments accumulate data.

## Running

```bash
# Single experiment: synthesize and evaluate 100 programs
python -m research --mode=synthesize --n 100

# Evolutionary search
python -m research --mode=evolve --n 20

# Continuous autonomous research (Aria runs experiments back-to-back)
python -m research --mode=continuous

# Launch the dashboard
python -m research --mode=dashboard
```

## LLM Backend Configuration

Set environment variables to give Aria a brain. Zero config = rule-based fallback.

```bash
# Ollama (local)
export ARIA_LLM_BACKEND=ollama
export OLLAMA_MODEL=llama3

# Anthropic
export ARIA_LLM_BACKEND=anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
export ARIA_LLM_BACKEND=openai
export OPENAI_API_KEY=sk-...
```

## Dashboard

A React frontend for monitoring experiments in real time. Features:

- Aria's mood-reactive SVG avatar and status
- Experiment control panel (start/stop, configure parameters)
- Live feed of program evaluations via SSE
- Drill-down from experiment list to full experiment detail (stage funnel, failure analysis, all programs with memory/FLOPs/baseline columns)
- Program detail modal with computation graph DAG visualization, extended fingerprint radar chart (8 axes), CKA similarity bars, training curve chart, sandbox timing, FLOP count
- Cross-experiment trend charts (S1 pass rate, novelty, loss ratio over time)
- **Learning tab**: grammar weights evolution (default vs learned), op success rate table, learning log timeline, efficiency frontier scatter plot
- Lab notebook and insights panels

```bash
cd research/dashboard && npm start    # development
python -m research --mode=dashboard   # production (serves built React app)
```

## Module Structure

```
research/
├── scientist/           # AI scientist system
│   ├── persona.py       # Dr. Aria Nexus personality + LLM integration
│   ├── runner.py        # Experiment execution engine (threaded)
│   ├── notebook.py      # SQLite lab notebook (70+ metrics schema)
│   ├── analytics.py     # Learning feedback engine (grammar weight optimization)
│   ├── api.py           # Flask REST API + SSE + analytics endpoints
│   └── llm/             # Pluggable LLM backends (Ollama, Anthropic, OpenAI)
├── synthesis/           # Program synthesis engine
│   ├── primitives.py    # ~50 tensor operations
│   ├── grammar.py       # Grammar-based graph generator (adaptive weights)
│   ├── graph.py         # Computation graph representation
│   ├── compiler.py      # Graph -> PyTorch module
│   ├── validator.py     # Graph legality checking
│   └── serializer.py    # JSON serialization
├── mathspaces/          # Exotic mathematical spaces
│   ├── tropical.py      # Tropical geometry (min-plus algebra)
│   ├── hyperbolic.py    # Hyperbolic geometry
│   ├── clifford.py      # Clifford algebras
│   └── padic.py         # P-adic numbers
├── eval/                # Evaluation and sandboxing
│   ├── sandbox.py       # Safe model execution with timeouts
│   ├── fingerprint.py   # Behavioral fingerprinting (13 metrics)
│   ├── metrics.py       # Novelty scoring
│   ├── flops.py         # FLOP estimation per op type
│   └── baseline.py      # Cached transformer baseline comparison
├── training/            # Training program synthesis
├── search/              # Evolution + novelty search
├── dashboard/           # React frontend
├── evaluator.py         # Stage 0/1 evaluation (standalone)
├── explorer.py          # Orchestrator
└── __main__.py          # CLI entry point
```

## Tech Stack

- **Core:** Python, PyTorch 2.0+
- **Frontend:** React 18
- **API:** Flask, SSE
- **Storage:** SQLite (lab notebook, experiment tracking, baseline cache)
- **LLM Integration:** Ollama, Anthropic Claude, OpenAI (optional)
