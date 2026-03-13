# Frontier Micronization Gap Memo For Aria

## Executive Summary
Top teams are spending a meaningful share of their effort on making capable models smaller, cheaper, faster, and more tool-usable rather than only scaling the largest frontier systems. The dominant patterns are:

- compact model families with strong reasoning or coding performance
- quantization and low-latency deployment
- sparse or architecture-efficient designs
- specialist small models for tool use, function calling, or code
- distillation, synthetic data, and targeted post-training to preserve quality at low size

Aria already has more pre-experiment structure than a typical lab toolchain: structured hypotheses, hypothesis critique, preregistration, historical analytics, a background knowledge distiller, and digest-driven config nudges. But Aria is still mostly an internal-history scientist, not a live frontier-aware compact-model scientist.

The biggest gaps are:

1. Aria does not appear to ingest live external research or benchmark shifts before proposing experiments.
2. Aria does not have a dedicated compact-model objective function for edge, low-memory, quantized, or sparse deployment targets.
3. Aria does not route pre-experiment reasoning through specialist math, code, and scientific-planning lanes.
4. Aria’s strongest analytics are mostly post hoc or every-few-cycles, not a compact real-time preflight board.

Bottom line: Aria should incorporate real-time math, programming, and AI into experiment design before experiments are launched. Math is mandatory. Programming/tool use is strongly recommended. AI is mandatory, but should be routed through compact specialist roles rather than one generic model pass.

## Where Frontier Teams Are Spending Time
### Google
- Google is pushing Gemma as a lightweight family designed to run on a single accelerator or directly on devices, with official quantized variants, long context, and function calling. That is a clear investment in portable capability, not just large-model scale.
- Google’s FunctionGemma shows another priority: small, specialized edge models that handle tool execution locally and route only harder work upward.
- Inference: Google is treating compact models as active system components, not only distilled assistants.

Sources:
- [Gemma 3](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-3/)
- [FunctionGemma](https://blog.google/technology/developers/functiongemma/)

### Microsoft
- Microsoft’s Phi line is explicitly framed as small language models with strong reasoning, especially math, plus multimodal and mini variants.
- Microsoft highlights high-quality synthetic data, curated organic data, post-training, reasoning-focused optimization, and low-bit deployment on NPUs.
- Inference: Microsoft is spending time on making reasoning survive compression, not merely on shrinking generic chat models.

Sources:
- [Introducing Phi-4](https://techcommunity.microsoft.com/blog/aiplatformblog/introducing-phi-4-microsoft%E2%80%99s-newest-small-language-model-specializing-in-comple/4357090)
- [Next generation of the Phi family](https://azure.microsoft.com/en-us/blog/empowering-innovation-the-next-generation-of-the-phi-family/)
- [One year of Phi](https://azure.microsoft.com/en-us/blog/one-year-of-phi-small-language-models-making-big-leaps-in-ai/)

### Meta
- Meta has pushed both small on-device Llama 3.2 models and larger multimodal variants, with the official message that 1B and 3B models are optimized for smartphones and future wearables.
- Meta also routes harder math and coding questions to stronger models inside Meta AI.
- Inference: Meta is treating compact models as deployment primitives and stronger models as escalation layers.

Sources:
- [Meta Connect 2024 / Llama 3.2](https://about.fb.com/de/news/2024/09/meta-connect-2024/)
- [Meta AI multilingual update](https://about.fb.com/news/2024/07/meta-ai-is-now-multilingual-more-creative-and-smarter/)

### Mistral
- Mistral has focused heavily on low-latency “small” models that target the bulk of real workloads, not only headline frontier benchmarks.
- Mistral Small 3.1 emphasizes low-latency function calling, local deployment, multimodal support, and fine-tuning into subject-matter experts.
- Codestral Mamba shows active investment in non-transformer architectures for efficient long-context code use.
- Inference: Mistral is spending time on architecture choices that reduce latency and keep specialized capability local.

Sources:
- [Mistral Small 3.1](https://mistral.ai/news/mistral-small-3-1)
- [Mistral Small 3](https://mistral.ai/fr/news/mistral-small-3)
- [Codestral Mamba](https://mistral.ai/news/codestral-mamba)

### OpenAI
- OpenAI is clearly investing in smaller, cheaper reasoning and coding models such as o3-mini, GPT-4.1 mini, and GPT-4.1 nano.
- The emphasis is cost-efficient reasoning, configurable reasoning effort, long context, function calling, and tool-capable smaller models.
- Inference: frontier capability is increasingly tiered into large, mini, and nano classes with explicit cost/latency tradeoffs.

Sources:
- [GPT-4.1](https://openai.com/index/gpt-4-1/)
- [OpenAI o3-mini](https://openai.com/index/openai-o3-mini/)
- [Introducing o3 and o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/)

### Anthropic
- Anthropic’s Haiku and Sonnet lines show a similar pattern: fast, cheaper models with strong coding and tool use, plus explicit support for token-efficient tool use and agentic coding workflows.
- Claude Code is especially relevant because it packages programming capability as a tool-using agent rather than only a chat model.
- Inference: Anthropic is spending time on practical agent efficiency, not just raw model size.

Sources:
- [Claude 3.7 Sonnet and Claude Code](https://www.anthropic.com/news/claude-3-7-sonnet)
- [Claude 3 Haiku](https://www.anthropic.com/news/claude-3-haiku)
- [Anthropic API release notes](https://docs.anthropic.com/en/release-notes/api)

## What Aria Already Does
Aria already has several strong pre-experiment and meta-experiment features:

- Structured hypothesis generation and parsing, including prediction, reasoning, test method, success metric, confounders, and fallback fields.
- Preflight hypothesis critique before launch, checking testability, specificity, novelty, and feasibility.
- Preregistration with explicit hypothesis fields, analysis plan, falsification conditions, and confounder checklist.
- Historical analytics over internal experiment data: convergence profiles, architecture-family clustering, config effects, op synergies, campaign tracking, and math-space coverage.
- A background knowledge distiller that converts internal experiment history into a compact digest, then injects that digest back into mode selection and config nudges.
- Digest-driven overrides that can change config values or op weighting before the next run.
- Some code-action capability through `edit_file` and `spawn_agent`, but this appears to be a chat/control surface, not a required pre-experiment design stage.

Repo evidence:
- `research/scientist/persona.py`
- `research/scientist/persona_hypothesis.py`
- `research/scientist/preregistration.py`
- `research/scientist/runner/results_knowledge.py`
- `research/scientist/intelligence/distiller.py`
- `research/scientist/intelligence/analyzer.py`
- `research/scientist/analytics/`

## Gaps Relative To Frontier Micronization
| Frontier trend | Evidence | Current Aria behavior | Gap | Recommended response |
|---|---|---|---|---|
| Compact models are first-class products | Google Gemma, Microsoft Phi, Mistral Small, OpenAI mini/nano, Anthropic Haiku | Aria measures novelty, loss, throughput, and some efficiency, but does not appear to optimize explicitly for on-device or compact-model deployment classes before search | High | Add compact-model targets to experiment design: memory ceiling, quantization fitness, active-parameter budget, edge latency budget |
| Specialist small models are routed to tool use, coding, or function calling | FunctionGemma, o3-mini, Claude Code, Mistral Small 3.1 | Aria uses a generic LLM backend plus rule-based fallback; no evidence of specialist model routing for math vs code vs scientific planning | High | Add preflight role routing: planner, math checker, code/tool agent, and evidence reviewer |
| Distillation and synthetic data preserve capability at small size | Phi-4, Phi reasoning work, Google Gemma family evolution | Aria learns from internal experiment history, but there is no clear compact-teacher or synthetic-task pretraining layer for experiment design | Medium | Add synthetic preflight tasks for architecture reasoning, math sanity, and failure prediction |
| Compact analytics are embedded in the design loop | Tool-efficient APIs, bounded reasoning effort, local NPUs, low-bit deployment | Aria has strong analytics, but much of it is background or after-the-fact rather than a lightweight live board before launch | High | Miniaturize analytics into a fast preflight scorecard built from cached digest features, top config effects, op synergies, and recent failure priors |
| Compact systems escalate to larger models only when needed | Meta escalation for hard tasks, Google edge-to-large routing, OpenAI reasoning-effort control | Aria does not appear to have explicit escalation tiers for easy vs hard pre-experiment reasoning | Medium | Add bounded escalation: small local model first, then stronger model only on ambiguity, conflict, or high-value experiments |
| Efficient architectures are themselves a research target | Codestral Mamba, Mistral sparse families, Meta on-device Llama, Gemma quantized deployment | Aria can search across many ops and math spaces, but micronized architectures are not clearly a first-order campaign objective | High | Add a dedicated “micronization campaign” mode with bias toward sparse, low-rank, quantizable, MoE, and state-space candidates |

## How Advanced Analytics Should Be Miniaturized
Aria does not need less analytics. Aria needs smaller, faster analytics in the pre-experiment loop.

Recommended miniaturized analytics stack:

- Keep only the highest-yield internal signals in the launch path:
  - top config effects
  - top synergistic and anti-synergistic op pairs
  - recent failure signatures
  - compact Pareto-family summaries
  - latest digest recommendations
- Replace broad free-text reasoning where possible with tool-backed checks:
  - FLOP and parameter calculators
  - memory and latency estimators
  - compile-feasibility checks
  - novelty/confounder checklists
- Use small specialist models for bounded jobs:
  - summarize evidence
  - critique hypothesis
  - draft preregistration
  - generate go/no-go recommendation
- Escalate to a stronger model only when:
  - math checks disagree
  - code/tool verification fails
  - the proposed experiment is expensive
  - the candidate is near a promotion threshold

Inference from the sources: this is the same pattern frontier labs are using at model level. Capability is becoming hierarchical and routed, not monolithic.

## Does Aria Need Real-Time Math, Programming, And AI Before Experiments?
### Real-time math
Yes. This is mandatory.

Why:
- compact-model work is constrained by parameter, FLOP, memory, latency, and scaling tradeoffs
- Aria already critiques feasibility, but there is no clear evidence of a live mathematical constraint board before every experiment
- math should validate expected direction, scaling plausibility, compression target, and likely failure modes before compute is spent

What it should do:
- parameter/FLOP/memory estimation
- sparse/quantized feasibility checks
- scaling sanity checks
- novelty-confidence and confounder sanity checks

### Real-time programming
Yes, but as tool use rather than unconstrained code generation.

Why:
- compact-model teams increasingly bind reasoning to tool execution, function calling, or coding agents
- Aria has `edit_file` and `spawn_agent`, but those are not clearly integrated into mandatory experiment design
- pre-experiment programming should validate that a proposed architecture or evaluation path is actually executable and instrumented

What it should do:
- compile dry-runs
- schema and shape validation
- automatic benchmark harness selection
- small generated probes for routing, sparsity, quantization, and latency

### Real-time AI
Yes. Also mandatory.

Why:
- Aria already uses LLMs, but mostly as a generic scientist persona
- frontier practice is moving toward role-specific, cost-aware, tool-aware model routing

What it should do:
- planner model: propose experiment
- math model or tool lane: prove feasibility
- code/tool lane: verify executability
- evidence lane: review novelty/confounders/risk

## Recommendations
### Immediate
1. Add a pre-experiment “design board” that scores every proposed run on compactness, feasibility, novelty confidence, and expected deployment efficiency.
2. Add role-routed preflight reasoning: planner, math checker, code/tool verifier, evidence reviewer.
3. Add a dedicated micronization campaign objective so Aria explicitly searches for compact, sparse, quantizable, low-latency architectures.

### Near-term
4. Add live external research retrieval with provenance, cached summaries, and benchmark trend updates before campaign formulation.
5. Add compact surrogate models for failure risk, compactness score, and expected utility before expensive execution.
6. Add escalation control so most preflight work runs on small local models and only hard cases call larger models.

### What Not To Do
- Do not turn every experiment launch into a heavyweight long-context research essay.
- Do not rely on one generic LLM prompt to handle math, code verification, novelty logic, and campaign planning at once.
- Do not keep micronization as a side metric; make it a first-class campaign target.

## Final Assessment
Aria is already stronger than many internal AI-scientist prototypes on structure, memory, and internal analytics. The missing piece is not “more intelligence” in the abstract. The missing piece is frontier-style compact intelligence:

- smaller specialist reasoning loops
- tighter preflight analytics
- explicit compact-model objectives
- live external awareness
- real-time math and tool-backed programming before launch

That is the gap between Aria’s current experiment design and where top LLM teams are concentrating effort.
