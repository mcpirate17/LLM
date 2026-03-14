/**
 * Converts React Flow nodes and edges into the canonical workflow JSON format.
 *
 * @param {Array} nodes - React Flow nodes
 * @param {Array} edges - React Flow edges
 * @param {Object} [meta] - Optional workflow identity (workflow_id, name, metadata)
 * @returns {Object} - Workflow JSON
 */
export function buildWorkflowJson(nodes, edges, meta = {}) {
  const wf = {
    schema_version: 'workflow_graph.v1',
    workflow_id: meta.workflow_id || 'wf_' + Date.now().toString(36),
    name: meta.name || 'Untitled Workflow',
    nodes: nodes.map((n) => ({
      id: n.id,
      component_type:
        n.data?.componentId ||
        (n.data?.manifest?.category && n.data?.manifest?.id
          ? `${n.data.manifest.category}/${n.data.manifest.id}`
          : (n.data?.category && n.data?.label
            ? `${n.data.category}/${String(n.data.label).trim().toLowerCase().replace(/\s+/g, '_')}`
            : 'unknown')),
      params: n.data?.paramValues || {},
      ui_meta: { position: n.position },
    })),
    edges: edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      source_port: e.sourceHandle || 'y',
      target_port: e.targetHandle || 'x',
    })),
  };
  if (meta.metadata && Object.keys(meta.metadata).length > 0) {
    wf.metadata = meta.metadata;
  }
  return wf;
}
