/**
 * Geometry utilities for canvas operations.
 */

/**
 * Calculates the shortest distance from a point (x, y) to a line segment defined by (x1, y1) and (x2, y2).
 */
export function getDistanceToSegment(x, y, x1, y1, x2, y2) {
  const A = x - x1;
  const B = y - y1;
  const C = x2 - x1;
  const D = y2 - y1;

  const dot = A * C + B * D;
  const len_sq = C * C + D * D;
  let param = -1;
  if (len_sq !== 0) {
    param = dot / len_sq;
  }

  let xx, yy;

  if (param < 0) {
    xx = x1;
    yy = y1;
  } else if (param > 1) {
    xx = x2;
    yy = y2;
  } else {
    xx = x1 + param * C;
    yy = y1 + param * D;
  }

  const dx = x - xx;
  const dy = y - yy;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Finds the closest edge to a given point on the canvas.
 * @param {Object} point - { x, y } in flow coordinates
 * @param {Array} edges - Array of React Flow edges
 * @param {Array} nodes - Array of React Flow nodes
 * @returns {Object|null} - The closest edge and its distance, or null
 */
export function findClosestEdge(point, edges, nodes) {
  if (!edges || edges.length === 0) return null;

  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  let closestEdge = null;
  let minDistance = Infinity;

  for (const edge of edges) {
    const sourceNode = nodeMap.get(edge.source);
    const targetNode = nodeMap.get(edge.target);

    if (!sourceNode || !targetNode) continue;

    // Estimate handle positions
    // DesignerNode handles are at Top and Bottom.
    const sWidth = sourceNode.measured?.width || 170;
    const sHeight = sourceNode.measured?.height || 80;
    const tWidth = targetNode.measured?.width || 170;
    const tHeight = targetNode.measured?.height || 80;

    // Source handle (Bottom)
    const x1 = sourceNode.position.x + sWidth / 2;
    const y1 = sourceNode.position.y + sHeight;

    // Target handle (Top)
    const x2 = targetNode.position.x + tWidth / 2;
    const y2 = targetNode.position.y;

    const distance = getDistanceToSegment(point.x, point.y, x1, y1, x2, y2);
    
    if (distance < minDistance) {
      minDistance = distance;
      closestEdge = edge;
    }
  }

  return { edge: closestEdge, distance: minDistance };
}
