# LLM Research

A multi-faceted LLM research platform combining efficient model architecture design, automated architecture discovery via program synthesis, and an AI scientist system that autonomously explores the space of novel computation patterns.

## Projects

### HYDRA — Hybrid Dynamic Routing Architecture

The primary model architecture. A transformer that combines three forms of dynamic routing to maximize compute efficiency:

- **CCGQA** (Compressed Convolutional Grouped Query Attention) — performs attention in a compressed latent space with causal convolutions
- **MoD** (Mixture-of-Depths) — token-level routing that selectively applies computation only to tokens that need it
- **MoR** (Mixture-of-Recursions) — layer-level adaptive depth with recursion, varying computation depth per position
- **MoE** (Mixture-of-Experts) — expert-level routing with load balancing

Validated model sizes from 50M to 1B parameters. Trains on A100 GPUs with 8-bit Adam, gradient checkpointing, and `torch.compile`.

```bash
python trainer.py --model_size 500M --mode production \
  --batch_size 4 --grad_accum 8 --seq_len 1024 \
  --moe --moe_num_layers 6 --8bit_adam
```

See [`HYDRA/README.md`](HYDRA/README.md) for full architecture documentation.

### research/ — Architecture Exploration & Program Synthesis

An autonomous system for discovering novel computation patterns through grammar-based program synthesis, evaluated through a multi-stage funnel.

#### How It Works

1. **Morphological Box** defines a structured architecture search space across ~10 dimensions (token representation, weight storage, mixing strategies, activations, normalization, residual connections, positional encodings, etc.)

2. **Program Synthesis Grammar** generates computation graphs from ~50 tensor operation primitives spanning elementwise, reduction, linear algebra, frequency domain, and exotic mathematical spaces (tropical geometry, hyperbolic space, Clifford algebras, p-adic numbers)

3. **Multi-Stage Evaluation Funnel** screens candidates efficiently:
   - **Stage 0** — compilation + forward/backward pass (~seconds)
   - **Stage 0.5** — numerical stability check
   - **Stage 1** — micro-training: 500 steps of AdamW, must show loss reduction
   - **Novelty scoring** — behavioral fingerprinting against known architectures

4. **Dr. Aria Nexus** (AI scientist persona) orchestrates the process: formulates hypotheses, designs experiments, analyzes results, and iterates. Backed by a pluggable LLM (Ollama/Claude/OpenAI) for intelligent analysis, with rule-based fallback when no LLM is configured.

#### Running

```bash
# Single experiment: synthesize and evaluate 100 programs
python -m research --mode=synthesize --n 100

# Evolutionary search
python -m research --mode=evolve --n 20

# Morphological box exploration
python -m research --mode=explore --n 20

# Continuous autonomous research (Aria runs experiments back-to-back)
python -m research --mode=continuous

# Launch the dashboard
python -m research --mode=dashboard
```

#### Dashboard

A React frontend for monitoring experiments in real time. Features:

- Aria's mood-reactive SVG avatar and status
- Experiment control panel (start/stop, configure parameters)
- Live feed of program evaluations via SSE
- Drill-down from experiment list to full experiment detail (stage funnel, failure analysis, all programs)
- Program detail modal with computation graph DAG visualization, fingerprint radar chart, and stage pipeline
- Cross-experiment trend charts (S1 pass rate, novelty, loss ratio over time)
- Lab notebook and insights panels

```bash
cd research/dashboard && npm start    # development
python -m research --mode=dashboard   # production (serves built React app)
```

#### LLM Backend Configuration

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

#### Module Structure

```
research/
├── scientist/           # AI scientist system
│   ├── persona.py       # Dr. Aria Nexus personality + LLM integration
│   ├── runner.py        # Experiment execution engine (threaded)
│   ├── notebook.py      # SQLite lab notebook
│   ├── api.py           # Flask REST API + SSE
│   └── llm/             # Pluggable LLM backends
├── synthesis/           # Program synthesis engine
│   ├── primitives.py    # ~50 tensor operations
│   ├── grammar.py       # Grammar-based graph generator
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
│   ├── fingerprint.py   # Behavioral fingerprinting
│   └── metrics.py       # Novelty scoring
├── training/            # Training program synthesis
├── search/              # Evolution + novelty search
├── dashboard/           # React frontend
├── morphological_box.py # Architecture search space definition
├── arch_builder.py      # ArchSpec -> PyTorch model
├── evaluator.py         # Stage 0/1 evaluation
├── explorer.py          # Orchestrator
└── __main__.py          # CLI entry point
```

### AbstractMoE — Semantic Token Language Model

An archived experiment exploring language modeling with semantic tokens instead of BPE subwords. Compresses vocabulary from 50K BPE tokens to ~8,700 semantic tokens (e.g., `[V:TRAVEL:PAST]`, `[N:ANIMAL]`, `[A:SIZE:LARGE]`). 856M total parameters, 311M active per forward pass via top-2 MoE routing across 6+1 experts.

See [`AbstractMoE/README_ABSTRACT_MOE_FINAL.md`](AbstractMoE/README_ABSTRACT_MOE_FINAL.md) for details.

### LA3 — Linear Attention Backend (Deprecated)

Triton-based linear attention kernels, removed from HYDRA due to gradient spike issues. Code preserved for reference.

## Tech Stack

- **Core:** Python, PyTorch 2.0+, CUDA/Triton
- **Performance:** Flash-Attention 2, Liger kernels, bitsandbytes (8-bit optimizers), torch.compile
- **Data:** HuggingFace datasets, tiktoken
- **Frontend:** React 18, recharts
- **API:** Flask, SSE
- **Storage:** SQLite (lab notebooks, experiment tracking)
- **LLM Integration:** Ollama, Anthropic Claude, OpenAI (optional, for AI scientist)
