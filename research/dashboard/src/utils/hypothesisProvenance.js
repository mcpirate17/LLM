const HYPOTHESIS_SOURCE_LABELS = {
  llm_context: 'LLM + Context',
  structured_hypothesis: 'LLM Structured',
  rule_based_fallback: 'Rule-Based Recovery',
  rule_based: 'Rule-Based',
  user_input: 'User Input',
  runner_template: 'Runner Template',
};

export function hypothesisProvenanceLabel(sourceOrMetadata) {
  const source = typeof sourceOrMetadata === 'string'
    ? sourceOrMetadata
    : sourceOrMetadata?.source;
  return HYPOTHESIS_SOURCE_LABELS[source] || null;
}

