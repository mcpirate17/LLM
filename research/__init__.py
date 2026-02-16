"""
HYDRA Architecture Explorer + Program Synthesis Engine

Two modes of operation:
1. Morphological exploration (original): combine known nn.Modules
2. Program synthesis (new): generate novel computation from primitives

The synthesis engine generates programs from ~50 primitive tensor operations,
compiles them to PyTorch modules, and evaluates them through a multi-stage
funnel. Dr. Aria Nexus (AI scientist) manages the research process.
"""
