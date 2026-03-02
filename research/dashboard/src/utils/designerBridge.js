export function parseDesignerBridgeMessage(data) {
  if (!data || data.source !== 'aria_designer') {
    return { kind: 'ignore' };
  }

  if (data.type === 'graph-loaded') {
    return {
      kind: 'graph-loaded',
      graphInfo: {
        name: data.name,
        nodeCount: data.nodeCount,
        edgeCount: data.edgeCount,
      },
      payload: data,
    };
  }

  if (data.type === 'graph-load-error') {
    return {
      kind: 'graph-load-error',
      error: data.error || 'Aria Designer could not load the requested architecture.',
      payload: data,
    };
  }

  if (data.type === 'graph-changed') {
    return {
      kind: 'graph-changed',
      graphInfo: {
        nodeCount: data.nodeCount,
        edgeCount: data.edgeCount,
      },
      payload: data,
    };
  }

  if (data.type === 'graph-data') {
    return {
      kind: 'graph-data',
      payload: data,
    };
  }

  if (data.type === 'embedded-ready') {
    return {
      kind: 'embedded-ready',
      payload: data,
    };
  }

  return { kind: 'ignore' };
}
