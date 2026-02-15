import React, { useMemo } from 'react';

/**
 * GraphViewer — SVG DAG renderer for computation graphs.
 *
 * Performs topological-sort layered layout of computation graph
 * nodes and edges. Designed for 10-16 node graphs, no heavy deps needed.
 *
 * Accepts graph_json_parsed from the API which has:
 *   { nodes: { "0": {id, op_name, input_ids, ...}, ... }, input_node_id, output_node_id }
 * Derives edges from input_ids on each node.
 */

const NODE_W = 100;
const NODE_H = 32;
const LAYER_GAP = 60;
const NODE_GAP = 16;

const OP_COLORS = {
  input:      '#58a6ff',
  output:     '#3fb950',
  linear:     '#bc8cff',
  attention:  '#f0883e',
  norm:       '#d29922',
  activation: '#f85149',
  residual:   '#8b949e',
  default:    '#484f58',
};

function getOpColor(op) {
  const lower = (op || '').toLowerCase();
  for (const [key, color] of Object.entries(OP_COLORS)) {
    if (lower.includes(key)) return color;
  }
  return OP_COLORS.default;
}

/**
 * Normalize graph data from various formats into { nodes: [...], edges: [...] }
 */
function normalizeGraph(graph) {
  if (!graph) return null;

  let nodes = [];
  let edges = [];

  // Handle nodes as dict (keyed by ID) — the format from graph_json_parsed
  if (graph.nodes && !Array.isArray(graph.nodes)) {
    const nodeDict = graph.nodes;
    nodes = Object.values(nodeDict).map(n => ({
      id: String(n.id),
      op: n.op_name || n.op || n.type || 'unknown',
      is_input: n.is_input || false,
      is_output: n.is_output || false,
    }));

    // Derive edges from input_ids
    for (const n of Object.values(nodeDict)) {
      const targetId = String(n.id);
      if (n.input_ids && Array.isArray(n.input_ids)) {
        for (const srcId of n.input_ids) {
          edges.push({ from: String(srcId), to: targetId });
        }
      }
    }
  } else if (Array.isArray(graph.nodes)) {
    // Already an array format
    nodes = graph.nodes.map(n => ({
      id: String(n.id),
      op: n.op_name || n.op || n.type || 'unknown',
      is_input: n.is_input || false,
      is_output: n.is_output || false,
    }));
    edges = (graph.edges || []).map(e => ({
      from: String(e.from),
      to: String(e.to),
    }));
  }

  if (nodes.length === 0) return null;
  return { nodes, edges };
}

function topoSort(nodes, edges) {
  const adj = {};
  const inDeg = {};
  nodes.forEach(n => { adj[n.id] = []; inDeg[n.id] = 0; });
  edges.forEach(e => {
    if (adj[e.from]) adj[e.from].push(e.to);
    inDeg[e.to] = (inDeg[e.to] || 0) + 1;
  });

  const layers = [];
  let queue = nodes.filter(n => (inDeg[n.id] || 0) === 0).map(n => n.id);
  const visited = new Set();

  while (queue.length > 0) {
    layers.push([...queue]);
    const next = [];
    for (const id of queue) {
      visited.add(id);
      for (const child of (adj[id] || [])) {
        inDeg[child]--;
        if (inDeg[child] === 0 && !visited.has(child)) next.push(child);
      }
    }
    queue = next;
    // Safety: break if we've visited all nodes (prevents infinite loop on cycles)
    if (visited.size >= nodes.length) break;
  }

  // Add any unvisited nodes (cycles) to a final layer
  const remaining = nodes.filter(n => !visited.has(n.id)).map(n => n.id);
  if (remaining.length > 0) layers.push(remaining);

  return layers;
}

function GraphViewer({ graph }) {
  const layout = useMemo(() => {
    const normalized = normalizeGraph(graph);
    if (!normalized) return null;

    const { nodes, edges } = normalized;
    const nodeMap = {};
    nodes.forEach(n => { nodeMap[n.id] = n; });

    const layers = topoSort(nodes, edges);

    // Assign positions
    const positions = {};
    let maxLayerWidth = 0;
    layers.forEach((layer, li) => {
      maxLayerWidth = Math.max(maxLayerWidth, layer.length);
      layer.forEach((id, ni) => {
        positions[id] = {
          x: ni * (NODE_W + NODE_GAP),
          y: li * (NODE_H + LAYER_GAP),
        };
      });
    });

    // Center each layer
    layers.forEach(layer => {
      const totalW = layer.length * (NODE_W + NODE_GAP) - NODE_GAP;
      const maxW = maxLayerWidth * (NODE_W + NODE_GAP) - NODE_GAP;
      const offset = (maxW - totalW) / 2;
      layer.forEach(id => { if (positions[id]) positions[id].x += offset; });
    });

    const svgW = maxLayerWidth * (NODE_W + NODE_GAP) + 20;
    const svgH = layers.length * (NODE_H + LAYER_GAP) + 20;

    return { nodeMap, edges, positions, svgW, svgH };
  }, [graph]);

  if (!layout) {
    return (
      <div className="card" style={{ padding: 24, textAlign: 'center', color: 'var(--text-muted)' }}>
        No graph data available
      </div>
    );
  }

  const { nodeMap, edges, positions, svgW, svgH } = layout;

  return (
    <div style={{ overflow: 'auto', maxHeight: 500 }}>
      <svg width={svgW} height={svgH} style={{ display: 'block', margin: '0 auto' }}>
        <defs>
          <marker id="arrow" viewBox="0 0 10 6" refX="10" refY="3"
            markerWidth="8" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 3 L 0 6 z" fill="#484f58" />
          </marker>
        </defs>

        {/* Edges */}
        {edges.map((e, i) => {
          const from = positions[e.from];
          const to = positions[e.to];
          if (!from || !to) return null;
          return (
            <line key={i}
              x1={from.x + NODE_W / 2 + 10} y1={from.y + NODE_H + 10}
              x2={to.x + NODE_W / 2 + 10} y2={to.y + 10}
              stroke="#484f58" strokeWidth={1.5}
              markerEnd="url(#arrow)" />
          );
        })}

        {/* Nodes */}
        {Object.entries(positions).map(([id, pos]) => {
          const node = nodeMap[id];
          const label = node?.op || id;
          const color = getOpColor(label);
          return (
            <g key={id} transform={`translate(${pos.x + 10}, ${pos.y + 10})`}>
              <rect width={NODE_W} height={NODE_H} rx={4}
                fill="var(--bg-tertiary, #21262d)"
                stroke={color} strokeWidth={1.5} />
              <text x={NODE_W / 2} y={NODE_H / 2 + 1}
                textAnchor="middle" dominantBaseline="middle"
                fill={color} fontSize={11} fontFamily="monospace">
                {label.length > 14 ? label.slice(0, 12) + '..' : label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export default GraphViewer;
