import { describe, it, expect } from 'vitest';
import { isValidConnection } from '../utils/validation';
import { buildWorkflowJson } from '../utils/workflow';

describe('isValidConnection', () => {
  const mockNodes = [
    {
      id: 'n1',
      data: {
        manifest: {
          outputs: [{ name: 'y', dtype: 'tensor' }]
        }
      }
    },
    {
      id: 'n2',
      data: {
        manifest: {
          inputs: [{ name: 'x', dtype: 'tensor' }]
        }
      }
    },
    {
      id: 'n3',
      data: {
        manifest: {
          inputs: [{ name: 'x', dtype: 'dataset' }]
        }
      }
    }
  ];

  it('allows compatible tensor connections', () => {
    const conn = { source: 'n1', target: 'n2', sourceHandle: 'y', targetHandle: 'x' };
    expect(isValidConnection(conn, mockNodes)).toBe(true);
  });

  it('rejects mismatched types (tensor -> dataset)', () => {
    const conn = { source: 'n1', target: 'n3', sourceHandle: 'y', targetHandle: 'x' };
    expect(isValidConnection(conn, mockNodes)).toBe(false);
  });

  it('rejects self-connections', () => {
    const conn = { source: 'n1', target: 'n1', sourceHandle: 'y', targetHandle: 'x' };
    expect(isValidConnection(conn, mockNodes)).toBe(false);
  });

  it('rejects duplicate edges', () => {
    const conn = { source: 'n1', target: 'n2', sourceHandle: 'y', targetHandle: 'x' };
    const existingEdges = [
      { source: 'n1', target: 'n2', sourceHandle: 'y', targetHandle: 'x' }
    ];
    expect(isValidConnection(conn, mockNodes, existingEdges)).toBe(false);
  });

  it('rejects when input port already connected', () => {
    const conn = { source: 'n1', target: 'n2', sourceHandle: 'y', targetHandle: 'x' };
    // n2's input 'x' is already connected from some other node
    const existingEdges = [
      { source: 'n3', target: 'n2', sourceHandle: 'y', targetHandle: 'x' }
    ];
    expect(isValidConnection(conn, mockNodes, existingEdges)).toBe(false);
  });

  it('rejects edges that would create a cycle', () => {
    // n1 → n2 already exists, trying to add n2 → n1 creates a cycle
    const conn = { source: 'n2', target: 'n1', sourceHandle: 'y', targetHandle: 'x' };
    const nodes = [
      { id: 'n1', data: { manifest: { outputs: [{ name: 'y', dtype: 'tensor' }], inputs: [{ name: 'x', dtype: 'tensor' }] } } },
      { id: 'n2', data: { manifest: { outputs: [{ name: 'y', dtype: 'tensor' }], inputs: [{ name: 'x', dtype: 'tensor' }] } } },
    ];
    const existingEdges = [
      { source: 'n1', target: 'n2', sourceHandle: 'y', targetHandle: 'x' }
    ];
    expect(isValidConnection(conn, nodes, existingEdges)).toBe(false);
  });

  it('rejects edges that would create an indirect cycle', () => {
    // n1 → n2 → n3 exists, trying to add n3 → n1 creates a cycle
    const conn = { source: 'n3', target: 'n1', sourceHandle: 'y', targetHandle: 'x' };
    const nodes = [
      { id: 'n1', data: { manifest: { outputs: [{ name: 'y', dtype: 'tensor' }], inputs: [{ name: 'x', dtype: 'tensor' }] } } },
      { id: 'n2', data: { manifest: { outputs: [{ name: 'y', dtype: 'tensor' }], inputs: [{ name: 'x', dtype: 'tensor' }] } } },
      { id: 'n3', data: { manifest: { outputs: [{ name: 'y', dtype: 'tensor' }], inputs: [{ name: 'x', dtype: 'tensor' }] } } },
    ];
    const existingEdges = [
      { source: 'n1', target: 'n2', sourceHandle: 'y', targetHandle: 'x' },
      { source: 'n2', target: 'n3', sourceHandle: 'y', targetHandle: 'x' },
    ];
    expect(isValidConnection(conn, nodes, existingEdges)).toBe(false);
  });

  it('allows valid acyclic connection', () => {
    // n1 → n2 exists, adding n2 → n3 is fine
    const conn = { source: 'n2', target: 'n3', sourceHandle: 'y', targetHandle: 'x' };
    const nodes = [
      { id: 'n1', data: { manifest: { outputs: [{ name: 'y', dtype: 'tensor' }] } } },
      { id: 'n2', data: { manifest: { outputs: [{ name: 'y', dtype: 'tensor' }], inputs: [{ name: 'x', dtype: 'tensor' }] } } },
      { id: 'n3', data: { manifest: { inputs: [{ name: 'x', dtype: 'tensor' }] } } },
    ];
    const existingEdges = [
      { source: 'n1', target: 'n2', sourceHandle: 'y', targetHandle: 'x' },
    ];
    expect(isValidConnection(conn, nodes, existingEdges)).toBe(true);
  });

  it('rejects connecting into a graph_input node', () => {
    const conn = { source: 'n2', target: 'input1', sourceHandle: 'y', targetHandle: 'x' };
    const nodes = [
      { id: 'input1', data: { componentId: 'io/input', manifest: { inputs: [], outputs: [{ name: 'y', dtype: 'tensor' }] } } },
      { id: 'n2', data: { manifest: { outputs: [{ name: 'y', dtype: 'tensor' }] } } },
    ];
    expect(isValidConnection(conn, nodes, [])).toBe(false);
  });
});

describe('buildWorkflowJson', () => {
  it('correctly maps nodes and edges', () => {
    const nodes = [{ id: 'n1', component_type: 'relu', position: {x:0, y:0}, data: { componentId: 'math/relu', paramValues: {a: 1} } }];
    const edges = [{ id: 'e1', source: 'n1', target: 'n2', sourceHandle: 'out', targetHandle: 'in' }];
    
    const wf = buildWorkflowJson(nodes, edges);
    expect(wf.nodes[0].id).toBe('n1');
    expect(wf.nodes[0].component_type).toBe('math/relu');
    expect(wf.nodes[0].params.a).toBe(1);
    expect(wf.edges[0].source_port).toBe('out');
  });
});
