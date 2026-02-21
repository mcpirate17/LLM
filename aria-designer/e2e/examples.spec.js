import { test, expect } from '@playwright/test';
import { fitCanvasIfAvailable, clickFirstDesignerNode } from './utils/canvas.js';

const EXAMPLES = [
  { label: 'Simple Linear', value: '/examples/simple_linear.json' },
  { label: 'Tropical Attention', value: '/examples/tropical_attention.json' },
  { label: 'Tropical Block', value: '/examples/tropical_block.json' },
  { label: 'Transformer Mini', value: '/examples/transformer_mini.json' },
  { label: 'SSM Stack', value: '/examples/ssm_stack.json' },
  { label: 'Hybrid Attn+SSM+MoE', value: '/examples/hybrid_attn_ssm_moe.json' },
];

test.describe('Example workflows', () => {
  test('load each example, inspect properties, and verify compile/test responses', async ({ page }) => {
    await page.goto('/');

    const statusbar = page.locator('.statusbar');
    await expect(statusbar).toBeVisible({ timeout: 15000 });
    await expect(statusbar).toContainText(/components loaded|API offline/i, { timeout: 15000 });

    for (const ex of EXAMPLES) {
      await page.selectOption('.actions select', ex.value);
      await page.waitForTimeout(250);

      await fitCanvasIfAvailable(page);

      const nodes = page.locator('.designer-node');
      await expect(nodes.first()).toBeVisible({ timeout: 10000 });
      const nodeCount = await nodes.count();
      expect(nodeCount).toBeGreaterThan(0);

      await clickFirstDesignerNode(page);
      await expect(page.locator('.panel.right .props-name')).toBeVisible({ timeout: 10000 });
      await expect(page.locator('.panel.right .props-cat')).toBeVisible({ timeout: 10000 });

      await page.getByRole('button', { name: /Step 2: Compile/i }).click();
      await expect(statusbar).toContainText(/compiled|compilation|failed/i, { timeout: 15000 });

      await page.getByRole('button', { name: /Step 3: Test/i }).click();
      await expect(statusbar).toContainText(/preview|success|failed|run complete/i, { timeout: 15000 });
    }
  });
});
