// Default port definitions for starter nodes
const tensorIn = [{ name: 'x', dtype: 'tensor' }]
const tensorOut = [{ name: 'y', dtype: 'tensor' }]
const binaryIn = [
  { name: 'a', dtype: 'tensor' },
  { name: 'b', dtype: 'tensor' },
]

export const palette = [
  { id: 'input', label: 'Input', category: 'io' },
  { id: 'linear_proj', label: 'Linear Projection', category: 'linear_algebra' },
  { id: 'split', label: 'Split', category: 'structural' },
  { id: 'sin', label: 'Sin', category: 'math' },
  { id: 'silu', label: 'SiLU', category: 'math' },
  { id: 'conv1d_seq', label: 'Conv1D Seq', category: 'sequence' },
  { id: 'concat', label: 'Concat', category: 'structural' },
  { id: 'add', label: 'Add', category: 'math' },
]

export const starterNodes = [
  {
    id: 'n1', type: 'designer', position: { x: 300, y: 40 },
    data: { label: 'Input', category: 'io', componentId: 'input', inputs: [], outputs: tensorOut },
  },
  {
    id: 'n2', type: 'designer', position: { x: 300, y: 150 },
    data: { label: 'Linear Proj', category: 'linear_algebra', componentId: 'linear_proj', inputs: tensorIn, outputs: tensorOut },
  },
  {
    id: 'n3', type: 'designer', position: { x: 300, y: 260 },
    data: { label: 'SiLU', category: 'math', componentId: 'silu', inputs: tensorIn, outputs: tensorOut },
  },
  {
    id: 'n4', type: 'designer', position: { x: 300, y: 370 },
    data: { label: 'Output', category: 'io', componentId: 'output_head', inputs: tensorIn, outputs: [{ name: 'logits', dtype: 'tensor' }] },
  },
]

export const starterEdges = [
  { id: 'e1-2', source: 'n1', target: 'n2', sourceHandle: 'y', targetHandle: 'x', },
  { id: 'e2-3', source: 'n2', target: 'n3', sourceHandle: 'y', targetHandle: 'x', },
  { id: 'e3-4', source: 'n3', target: 'n4', sourceHandle: 'y', targetHandle: 'x', },
]
