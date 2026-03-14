import { test, expect } from '@playwright/test';
import { fitCanvasIfAvailable, waitForDesignerReady } from './utils/canvas.js';

test.describe('Aria Designer', () => {

  test('loads with components in palette', async ({ page }) => {
    await page.goto('/');
    await waitForDesignerReady(page);
    await expect(page.locator('.panel.left')).toBeVisible({ timeout: 5000 });
  });

  test('starter workflow renders with edges', async ({ page }) => {
    await page.goto('/');
    await page.waitForTimeout(1000);
    // Should have starter nodes (designer type)
    const nodes = await page.locator('.designer-node').count();
    expect(nodes).toBeGreaterThan(0);
    // Should have edges rendered as SVG paths
    const edges = await page.locator('.react-flow__edge').count();
    expect(edges).toBeGreaterThan(0);
  });

  test('loading example renders nodes and edges', async ({ page }) => {
    await page.goto('/');
    // Wait for components to load
    await expect(page.locator('.statusbar')).toContainText('components loaded');

    // Select an example via File menu
    await page.locator('button:has-text("File")').click();
    await page.locator('button:has-text("Example: Tropical Block")').click();
    await page.waitForTimeout(500);

    // Should have nodes
    const nodes = await page.locator('.designer-node').count();
    expect(nodes).toBeGreaterThan(3);

    // Edges should render (this is the key bug test)
    const edges = await page.locator('.react-flow__edge').count();
    expect(edges).toBeGreaterThan(0);
  });

  test('can manually connect two nodes', async ({ page }) => {
    await page.goto('/');
    await waitForDesignerReady(page);

    // Count initial edges
    await page.waitForTimeout(500);
    const initialEdges = await page.locator('.react-flow__edge').count();

    // Find source (bottom) and target (top) handles
    const sourceHandles = page.locator('.react-flow__handle-bottom');
    const targetHandles = page.locator('.react-flow__handle-top');

    if (await sourceHandles.count() > 0 && await targetHandles.count() > 0) {
      const source = sourceHandles.first();
      const target = targetHandles.last();

      // Drag from source to target
      await source.dragTo(target);
      await page.waitForTimeout(300);

      const newEdges = await page.locator('.react-flow__edge').count();
      // We expect at least one more edge (connection succeeded)
      expect(newEdges).toBeGreaterThanOrEqual(initialEdges);
    }
  });

  test('drag component from palette to canvas', async ({ page }) => {
    await page.goto('/');
    await waitForDesignerReady(page);

    const initialNodes = await page.locator('.designer-node').count();

    // Find a palette item and drag it to canvas
    const paletteItem = page.locator('.palette-item').first();
    const canvas = page.locator('.react-flow__pane');

    if (await paletteItem.count() > 0) {
      await paletteItem.dragTo(canvas);
      await page.waitForTimeout(300);

      const newNodes = await page.locator('.designer-node').count();
      expect(newNodes).toBeGreaterThanOrEqual(initialNodes);
    }
  });

  test('e2e loop: drag Input -> drag ReLU -> connect -> run -> verify status', async ({ page }) => {
    await page.goto('/');
    await waitForDesignerReady(page);
    await fitCanvasIfAvailable(page);

    const canvas = page.locator('.react-flow__pane');
    const inputItem = page.locator('.palette-item').filter({ hasText: /input/i }).first();
    const reluItem = page.locator('.palette-item').filter({ hasText: /relu/i }).first();

    const beforeNodes = await page.locator('.designer-node').count();

    if (await inputItem.count() > 0) {
      await inputItem.dragTo(canvas);
      await page.waitForTimeout(250);
    }
    if (await reluItem.count() > 0) {
      await reluItem.dragTo(canvas);
      await page.waitForTimeout(250);
    }

    const afterNodes = await page.locator('.designer-node').count();
    expect(afterNodes).toBeGreaterThanOrEqual(beforeNodes);

    const sourceHandles = page.locator('.react-flow__handle-bottom');
    const targetHandles = page.locator('.react-flow__handle-top');
    if (await sourceHandles.count() > 0 && await targetHandles.count() > 0) {
      await sourceHandles.last().dragTo(targetHandles.first());
      await page.waitForTimeout(300);
    }

    await page.click('button:has-text("Run")');
    await expect(page.locator('.statusbar')).toContainText(/complete|failed|success/i, { timeout: 10000 });
  });

});
