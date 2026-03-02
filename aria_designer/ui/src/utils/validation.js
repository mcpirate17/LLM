/**
 * Validates a connection between two nodes in React Flow.
 * Prevents illogical component ordering and structural issues at wire-time.
 *
 * @param {Object} connection - React Flow connection object {source, target, sourceHandle, targetHandle}
 * @param {Array} nodes - Current list of React Flow nodes
 * @param {Array} edges - Current list of React Flow edges
 * @returns {boolean} - True if connection is valid
 */
export const isValidConnection = (connection, nodes, edges = []) => {
  if (!connection.source || !connection.target) return false

  // 1. No self-connections
  if (connection.source === connection.target) return false

  const sourceNode = nodes.find(n => n.id === connection.source)
  const targetNode = nodes.find(n => n.id === connection.target)

  if (!sourceNode || !targetNode) return false

  // 2. Port dtype compatibility
  const sourcePort = sourceNode.data?.manifest?.outputs?.find(
    p => p.name === (connection.sourceHandle || 'y')
  )
  const targetPort = targetNode.data?.manifest?.inputs?.find(
    p => p.name === (connection.targetHandle || 'x')
  )

  if (sourcePort && targetPort && sourcePort.dtype !== targetPort.dtype) {
    return false
  }

  // 3. No duplicate edges (same source port → same target port)
  const srcHandle = connection.sourceHandle || 'y'
  const tgtHandle = connection.targetHandle || 'x'
  const isDuplicate = edges.some(
    e => e.source === connection.source &&
         e.target === connection.target &&
         (e.sourceHandle || 'y') === srcHandle &&
         (e.targetHandle || 'x') === tgtHandle
  )
  if (isDuplicate) return false

  // 4. Single-connection per input port — each input port accepts at most one edge
  //    (use a merge/concat node to combine multiple sources)
  const inputAlreadyConnected = edges.some(
    e => e.target === connection.target &&
         (e.targetHandle || 'x') === tgtHandle
  )
  if (inputAlreadyConnected) return false

  // 5. Source/sink role enforcement
  const srcType = _componentType(sourceNode)
  const tgtType = _componentType(targetNode)

  // graph_input/input nodes are sources — nothing should feed into them
  // (already enforced by having no input handles, but double-check)
  if (_isGraphInput(tgtType) && (targetNode.data?.manifest?.inputs?.length || 0) === 0) {
    return false
  }

  // 6. Cycle detection — would this edge create a cycle?
  if (_wouldCreateCycle(connection.source, connection.target, nodes, edges)) {
    return false
  }

  return true
}

/**
 * Extract the component type identifier from a node.
 */
function _componentType(node) {
  return node.data?.componentId || node.data?.component_type || ''
}

/**
 * Check if a component type is a graph input.
 */
function _isGraphInput(type) {
  const base = type.split('/').pop()
  return base === 'input' || base === 'graph_input'
}

/**
 * Check if adding an edge source→target would create a cycle.
 * Uses DFS from target following existing edges to see if we can reach source.
 */
function _wouldCreateCycle(sourceId, targetId, nodes, edges) {
  // If we can reach sourceId by following edges forward from targetId,
  // then adding source→target creates a cycle.
  const adj = {}
  for (const e of edges) {
    if (!adj[e.source]) adj[e.source] = []
    adj[e.source].push(e.target)
  }

  const visited = new Set()
  const stack = [targetId]

  while (stack.length > 0) {
    const current = stack.pop()
    if (current === sourceId) return true
    if (visited.has(current)) continue
    visited.add(current)
    const neighbors = adj[current]
    if (neighbors) {
      for (const n of neighbors) {
        stack.push(n)
      }
    }
  }

  return false
}
