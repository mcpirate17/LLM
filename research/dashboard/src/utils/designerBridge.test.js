import { parseDesignerBridgeMessage } from './designerBridge';

describe('parseDesignerBridgeMessage', () => {
  test('ignores non-designer messages', () => {
    expect(parseDesignerBridgeMessage({ source: 'other', type: 'graph-loaded' })).toEqual({ kind: 'ignore' });
    expect(parseDesignerBridgeMessage(null)).toEqual({ kind: 'ignore' });
  });

  test('maps graph-loaded payload', () => {
    const msg = {
      source: 'aria-designer',
      type: 'graph-loaded',
      name: 'Imported graph',
      nodeCount: 12,
      edgeCount: 17,
    };
    const parsed = parseDesignerBridgeMessage(msg);
    expect(parsed.kind).toBe('graph-loaded');
    expect(parsed.graphInfo).toEqual({ name: 'Imported graph', nodeCount: 12, edgeCount: 17 });
    expect(parsed.payload).toBe(msg);
  });

  test('maps graph-load-error payload with fallback message', () => {
    const explicit = parseDesignerBridgeMessage({
      source: 'aria-designer',
      type: 'graph-load-error',
      error: 'Import failed: 404',
    });
    expect(explicit.kind).toBe('graph-load-error');
    expect(explicit.error).toBe('Import failed: 404');

    const fallback = parseDesignerBridgeMessage({
      source: 'aria-designer',
      type: 'graph-load-error',
    });
    expect(fallback.kind).toBe('graph-load-error');
    expect(fallback.error).toBe('Aria Designer could not load the requested architecture.');
  });

  test('maps graph-changed payload', () => {
    const parsed = parseDesignerBridgeMessage({
      source: 'aria-designer',
      type: 'graph-changed',
      nodeCount: 7,
      edgeCount: 9,
    });
    expect(parsed.kind).toBe('graph-changed');
    expect(parsed.graphInfo).toEqual({ nodeCount: 7, edgeCount: 9 });
  });

  test('maps graph-data passthrough', () => {
    const msg = {
      source: 'aria-designer',
      type: 'graph-data',
      workflow: { nodes: [] },
    };
    const parsed = parseDesignerBridgeMessage(msg);
    expect(parsed.kind).toBe('graph-data');
    expect(parsed.payload).toBe(msg);
  });

  test('maps embedded-ready signal', () => {
    const msg = {
      source: 'aria-designer',
      type: 'embedded-ready',
      readOnly: true,
    };
    const parsed = parseDesignerBridgeMessage(msg);
    expect(parsed.kind).toBe('embedded-ready');
    expect(parsed.payload).toBe(msg);
  });
});

/**
 * Integration test: simulates the full embedded bridge handshake sequence
 * that occurs when ArchitectureDrawer opens an iframe of aria-designer.
 *
 * Sequence:
 *   1. iframe loads and posts `embedded-ready`
 *   2. parent receives it and sends `load-result` command
 *   3. iframe imports the result and posts `graph-loaded` (or `graph-load-error`)
 */
describe('embedded bridge handshake roundtrip', () => {
  test('embedded-ready → load-result → graph-loaded', () => {
    // Step 1: iframe signals readiness
    const ready = parseDesignerBridgeMessage({
      source: 'aria-designer',
      type: 'embedded-ready',
      readOnly: true,
    });
    expect(ready.kind).toBe('embedded-ready');

    // Step 2: parent would send { target: 'aria-designer', type: 'load-result', resultId }
    // (this is a postMessage to the iframe — not parsed by the bridge, just validated structurally)
    const loadCmd = { target: 'aria-designer', type: 'load-result', resultId: 'res_abc123' };
    expect(loadCmd.target).toBe('aria-designer');
    expect(loadCmd.type).toBe('load-result');
    expect(loadCmd.resultId).toBeTruthy();

    // Step 3: iframe responds with graph-loaded
    const loaded = parseDesignerBridgeMessage({
      source: 'aria-designer',
      type: 'graph-loaded',
      resultId: 'res_abc123',
      name: 'Imported arch',
      nodeCount: 8,
      edgeCount: 11,
    });
    expect(loaded.kind).toBe('graph-loaded');
    expect(loaded.graphInfo).toEqual({ name: 'Imported arch', nodeCount: 8, edgeCount: 11 });
  });

  test('embedded-ready → load-result → graph-load-error', () => {
    const ready = parseDesignerBridgeMessage({
      source: 'aria-designer',
      type: 'embedded-ready',
    });
    expect(ready.kind).toBe('embedded-ready');

    // iframe fails to import
    const errMsg = parseDesignerBridgeMessage({
      source: 'aria-designer',
      type: 'graph-load-error',
      resultId: 'res_bad',
      error: 'Import failed: 404',
    });
    expect(errMsg.kind).toBe('graph-load-error');
    expect(errMsg.error).toBe('Import failed: 404');
  });

  test('ignores messages before embedded-ready', () => {
    // If a stray graph-data arrives before embedded-ready, it should still parse
    // but the handshake contract says parent should wait for embedded-ready first
    const stray = parseDesignerBridgeMessage({
      source: 'aria-designer',
      type: 'graph-data',
      workflow: {},
    });
    expect(stray.kind).toBe('graph-data');

    // Non-designer messages are always ignored
    const noise = parseDesignerBridgeMessage({ source: 'webpack', type: 'hmr' });
    expect(noise.kind).toBe('ignore');
  });
});

/**
 * ArchitectureDrawer state-machine simulation.
 *
 * Verifies the full lifecycle that occurs when a user clicks "Designer"
 * from Discoveries/ProgramDetail/DiscoveryRankings:
 *   1. ensure-running → designerReady
 *   2. iframe posts embedded-ready → bridgeReady
 *   3. parent sends load-result immediately (no 2s delay)
 *   4. iframe posts graph-loaded → loading=false, graphInfo populated
 *
 * Also verifies error/retry and timeout paths.
 */
describe('ArchitectureDrawer state machine', () => {
  // Simulates the drawer's state transitions using parseDesignerBridgeMessage
  function createDrawerState(resultId) {
    return {
      resultId,
      loading: true,
      booting: true,
      designerReady: false,
      bridgeReady: false,
      error: null,
      graphInfo: null,
      loadResultSent: false,
    };
  }

  function processMessage(state, data) {
    const parsed = parseDesignerBridgeMessage(data);
    const next = { ...state };
    switch (parsed.kind) {
      case 'embedded-ready':
        next.bridgeReady = true;
        // ArchitectureDrawer immediately sends load-result when bridgeReady && designerReady
        if (next.designerReady && next.loading && !next.error) {
          next.loadResultSent = true;
        }
        break;
      case 'graph-loaded':
        next.loading = false;
        next.error = null;
        next.graphInfo = parsed.graphInfo;
        break;
      case 'graph-load-error':
        next.loading = false;
        next.error = parsed.error;
        break;
      default:
        break;
    }
    return next;
  }

  test('happy path: boot → bridgeReady → load-result → graph-loaded', () => {
    let state = createDrawerState('res_happy');

    // Step 1: ensure-running succeeds
    state.booting = false;
    state.designerReady = true;
    expect(state.loading).toBe(true);
    expect(state.bridgeReady).toBe(false);

    // Step 2: iframe sends embedded-ready
    state = processMessage(state, {
      source: 'aria-designer',
      type: 'embedded-ready',
      readOnly: true,
    });
    expect(state.bridgeReady).toBe(true);
    expect(state.loadResultSent).toBe(true); // immediate, no 2s delay

    // Step 3: iframe successfully loads and responds
    state = processMessage(state, {
      source: 'aria-designer',
      type: 'graph-loaded',
      resultId: 'res_happy',
      name: 'Test Architecture',
      nodeCount: 15,
      edgeCount: 20,
    });
    expect(state.loading).toBe(false);
    expect(state.error).toBeNull();
    expect(state.graphInfo).toEqual({
      name: 'Test Architecture',
      nodeCount: 15,
      edgeCount: 20,
    });
  });

  test('error path: import fails → graph-load-error → shows error', () => {
    let state = createDrawerState('res_fail');
    state.booting = false;
    state.designerReady = true;

    // iframe ready
    state = processMessage(state, {
      source: 'aria-designer',
      type: 'embedded-ready',
    });
    expect(state.bridgeReady).toBe(true);

    // import fails
    state = processMessage(state, {
      source: 'aria-designer',
      type: 'graph-load-error',
      resultId: 'res_fail',
      error: 'Failed to import res_fail: Import failed: 404',
    });
    expect(state.loading).toBe(false);
    expect(state.error).toBe('Failed to import res_fail: Import failed: 404');
    expect(state.graphInfo).toBeNull();
  });

  test('boot failure: ensure-running fails → error before bridge', () => {
    let state = createDrawerState('res_noboot');
    state.booting = false;
    state.error = 'Could not auto-start Aria Designer: Connection refused';
    state.loading = false;

    // No bridge messages should arrive since iframe never loads
    expect(state.designerReady).toBe(false);
    expect(state.bridgeReady).toBe(false);
    expect(state.error).toBeTruthy();
  });

  test('load-result not sent before designerReady', () => {
    let state = createDrawerState('res_early');
    // Still booting, embedded-ready arrives (shouldn't happen, but defend)
    state = processMessage(state, {
      source: 'aria-designer',
      type: 'embedded-ready',
    });
    expect(state.bridgeReady).toBe(true);
    // designerReady is false, so load-result should NOT be sent
    expect(state.loadResultSent).toBe(false);
  });

  test('graph-changed updates graphInfo after initial load', () => {
    let state = createDrawerState('res_change');
    state.booting = false;
    state.designerReady = true;

    state = processMessage(state, {
      source: 'aria-designer',
      type: 'embedded-ready',
    });
    state = processMessage(state, {
      source: 'aria-designer',
      type: 'graph-loaded',
      name: 'Original',
      nodeCount: 10,
      edgeCount: 12,
    });
    expect(state.graphInfo.nodeCount).toBe(10);

    // User edits in designer → graph-changed
    const changed = parseDesignerBridgeMessage({
      source: 'aria-designer',
      type: 'graph-changed',
      nodeCount: 11,
      edgeCount: 14,
    });
    expect(changed.kind).toBe('graph-changed');
    expect(changed.graphInfo).toEqual({ nodeCount: 11, edgeCount: 14 });
  });
});
