/**
 * Validates a connection between two nodes in React Flow.
 * 
 * @param {Object} connection - React Flow connection object {source, target, sourceHandle, targetHandle}
 * @param {Array} nodes - Current list of React Flow nodes
 * @returns {boolean} - True if connection is valid
 */
export const isValidConnection = (connection, nodes) => {
  if (!connection.source || !connection.target) return false;
  
  // Self-connection check
  if (connection.source === connection.target) return false;

  const sourceNode = nodes.find(n => n.id === connection.source);
  const targetNode = nodes.find(n => n.id === connection.target);
  
  if (!sourceNode || !targetNode) return false;

  // Type check
  const sourcePort = sourceNode.data?.manifest?.outputs?.find(p => p.name === (connection.sourceHandle || 'y'));
  const targetPort = targetNode.data?.manifest?.inputs?.find(p => p.name === (connection.targetHandle || 'x'));

  if (sourcePort && targetPort) {
    if (sourcePort.dtype !== targetPort.dtype) {
      return false;
    }
  }
  
  return true;
};
