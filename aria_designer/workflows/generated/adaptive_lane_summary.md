# Adaptive Lane Workflow Summary

Requested pattern: input -> difficulty scoring -> conditional split into fast/hard lanes -> GELU on both lanes -> combine -> compress -> output.

Inspired by survivor fingerprint `23a9c75ef7ee1da6`, mapped onto approved Aria Designer components.

## Local Preview Ranking

1. `wf_adaptive_lane_moe_v1` - `None` ms preview latency
   - Adaptive Lane Split V1 - Fast GELU vs MoE
   - Validate: `True`
2. `wf_adaptive_lane_moe_v2` - `None` ms preview latency
   - Adaptive Lane Split V2 - Ultrametric Routed-MoE
   - Validate: `True`
3. `wf_adaptive_lane_moe_v3` - `None` ms preview latency
   - Adaptive Lane Split V3 - Routed-MoE with Dense Refinement
   - Validate: `True`
4. `wf_adaptive_lane_moe_v4` - `None` ms preview latency
   - Adaptive Lane Split V4 - Ultrametric Routed-MoE Bottleneck
   - Validate: `True`

## Direct Bridge Evaluation Ranking

1. `wf_adaptive_lane_moe_v4` - `734.4167349947384` ms total bridge eval
   - structural novelty `0.4064401090145111`
   - behavioral novelty `0.6294807854552434`
   - overall novelty `0.395332932472229`
   - fingerprint `d8266ee322fd7cc6`
2. `wf_adaptive_lane_moe_v3` - `734.889352999744` ms total bridge eval
   - structural novelty `0.32677000761032104`
   - behavioral novelty `0.611336552722989`
   - overall novelty `0.3731329143047333`
   - fingerprint `abc886a3994c5238`
3. `wf_adaptive_lane_moe_v2` - `740.5487280047964` ms total bridge eval
   - structural novelty `0.4006969630718231`
   - behavioral novelty `0.6366936924097744`
   - overall novelty `0.40234750509262085`
   - fingerprint `0e9b25e75cd5e12e`
4. `wf_adaptive_lane_moe_v1` - `1381.6298379970249` ms total bridge eval
   - structural novelty `0.32095298171043396`
   - behavioral novelty `0.6062185052866104`
   - overall novelty `0.3674008846282959`
   - fingerprint `89387bfc1d302246`

Bridge novelty fix: the runtime bridge now passes the computed behavioral fingerprint into `novelty_score(...)`, so advanced routed graphs no longer get scored as structural-only during evaluation.
