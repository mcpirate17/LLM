export const DESIGNER_BASE = process.env.REACT_APP_DESIGNER_URL || '/designer-proxy';

export const INTENTS = [
  { key: 'balanced', label: 'Balanced', color: 'var(--text-secondary)' },
  { key: 'quality', label: 'Quality', color: 'var(--accent-green)' },
  { key: 'compression', label: 'Compression', color: 'var(--accent-blue)' },
  { key: 'sparsity', label: 'Sparsity', color: 'var(--accent-purple)' },
  { key: 'novelty', label: 'Novelty', color: 'var(--accent-orange, #e88d3f)' },
];

export function toIntId(value) {
  const num = Number(value);
  return Number.isFinite(num) ? String(Math.trunc(num)) : String(value);
}

export function analyzeResearchGraph(graphJson) {
  if (!graphJson || typeof graphJson !== 'object') return null;
  const nodesObj = graphJson.nodes || {};
  const nodeIds = Object.keys(nodesObj);
  if (!nodeIds.length) return null;

  const outputNodeId = graphJson.output_node_id != null
    ? toIntId(graphJson.output_node_id)
    : (nodeIds.find((id) => nodesObj[id]?.is_output) || null);
  const inputNodeId = graphJson.input_node_id != null
    ? toIntId(graphJson.input_node_id)
    : (nodeIds.find((id) => nodesObj[id]?.is_input) || null);
  if (!outputNodeId || !nodesObj[outputNodeId]) return null;

  const reachable = new Set();
  const stack = [outputNodeId];
  while (stack.length) {
    const current = stack.pop();
    if (current == null || reachable.has(current)) continue;
    reachable.add(current);
    const node = nodesObj[current];
    const inputs = Array.isArray(node?.input_ids) ? node.input_ids : [];
    for (const parent of inputs) {
      const parentId = toIntId(parent);
      if (!reachable.has(parentId) && nodesObj[parentId]) stack.push(parentId);
    }
  }

  let edgeCount = 0;
  for (const id of nodeIds) {
    const inputs = Array.isArray(nodesObj[id]?.input_ids) ? nodesObj[id].input_ids : [];
    edgeCount += inputs.length;
  }

  return {
    nodeCount: nodeIds.length,
    edgeCount,
    outputNodeId,
    inputNodeId,
    reachableCount: reachable.size,
    deadNodeCount: Math.max(0, nodeIds.length - reachable.size),
    hasInputPath: inputNodeId ? reachable.has(inputNodeId) : false,
  };
}

export function analyzeWorkflowGraph(workflow) {
  if (!workflow || typeof workflow !== 'object') return null;
  const nodes = Array.isArray(workflow.nodes) ? workflow.nodes : [];
  const edges = Array.isArray(workflow.edges) ? workflow.edges : [];
  if (!nodes.length) return null;

  const nodeById = new Map(nodes.map((n) => [String(n.id), n]));
  const incoming = new Map();
  for (const edge of edges) {
    const src = String(edge.source);
    const dst = String(edge.target);
    if (!incoming.has(dst)) incoming.set(dst, []);
    incoming.get(dst).push(src);
  }

  const isOutputType = (type) => {
    const t = String(type || '').toLowerCase();
    return t === 'graph_output' || t.endsWith('/output') || t === 'output' || t === 'io/output';
  };
  const isInputType = (type) => {
    const t = String(type || '').toLowerCase();
    return t === 'graph_input' || t.endsWith('/input') || t === 'input' || t === 'io/input';
  };

  const outputNode = nodes.find((n) => isOutputType(n.component_type)) || null;
  const outputNodeId = outputNode ? String(outputNode.id) : null;
  if (!outputNodeId) return null;

  const inputNodeIds = new Set(
    nodes.filter((n) => isInputType(n.component_type)).map((n) => String(n.id))
  );

  const reachable = new Set();
  const stack = [outputNodeId];
  while (stack.length) {
    const current = stack.pop();
    if (!current || reachable.has(current)) continue;
    reachable.add(current);
    const parents = incoming.get(current) || [];
    for (const parent of parents) {
      if (!reachable.has(parent) && nodeById.has(parent)) stack.push(parent);
    }
  }

  let hasInputPath = false;
  for (const nid of inputNodeIds) {
    if (reachable.has(nid)) {
      hasInputPath = true;
      break;
    }
  }

  return {
    nodeCount: nodes.length,
    edgeCount: edges.length,
    outputNodeId,
    inputNodeCount: inputNodeIds.size,
    reachableCount: reachable.size,
    deadNodeCount: Math.max(0, nodes.length - reachable.size),
    hasInputPath,
  };
}

export function buildIntegrityWarning(sourceCheck, designerCheck) {
  if (!sourceCheck || !designerCheck) return null;

  const issues = [];
  if (sourceCheck.nodeCount !== designerCheck.nodeCount) {
    issues.push(
      `node count mismatch (backend \${sourceCheck.nodeCount}, viewer \${designerCheck.nodeCount})`
    );
  }
  if (sourceCheck.edgeCount !== designerCheck.edgeCount) {
    issues.push(
      `edge count mismatch (backend \${sourceCheck.edgeCount}, viewer \${designerCheck.edgeCount})`
    );
  }
  if (sourceCheck.hasInputPath && !designerCheck.hasInputPath) {
    issues.push('viewer graph appears disconnected from input to output');
  }
  if (sourceCheck.deadNodeCount === 0 && designerCheck.deadNodeCount > 0) {
    issues.push(`viewer has \${designerCheck.deadNodeCount} unreachable node(s)`);
  }
  if (!issues.length) return null;
  return `Graph integrity mismatch: \${issues.join('; ')}. Backend-tested graph remains authoritative.`;
}
