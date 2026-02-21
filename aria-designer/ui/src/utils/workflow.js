/**
 * Converts React Flow nodes and edges into the canonical workflow JSON format.
 * 
 * @param {Array} nodes - React Flow nodes
 * @param {Array} edges - React Flow edges
 * @returns {Object} - Workflow JSON
 */
export function buildWorkflowJson(nodes, edges) {
  return {
    schema_version: 'workflow_graph.v1',
    workflow_id: 'wf_' + Date.now().toString(36),
    name: 'Untitled Workflow',
    nodes: nodes.map((n) => ({
      id: n.id,
      component_type: n.data?.componentId || n.data?.label || 'unknown',
      params: n.data?.paramValues || {},
      ui_meta: { position: n.position },
    })),
    edges: edges.map((e) => ({
      id: e.id,
      source: e.source,
      source_port: e.sourceHandle || 'y',
      target: e.target,
      target_port: e.targetHandle || 'x',
    })),
  };
}
