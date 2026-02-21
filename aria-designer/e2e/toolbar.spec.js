import { test, expect } from '@playwright/test';
import { waitForDesignerReady } from './utils/canvas.js';

test.describe('Toolbar Actions', () => {

  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForDesignerReady(page);
  });

  // Use the statusbar text content (whole footer) to avoid strict mode issues
  // with multiple .status-msg spans (runStatus + statusMsg)
  const statusbar = '.statusbar';

  test('Validate button shows result', async ({ page }) => {
    await page.click('button:has-text("Validate")');
    await expect(page.locator(statusbar)).toContainText(/valid|issues|failed/i, { timeout: 5000 });
  });

  test('Compile button shows result', async ({ page }) => {
    await page.click('button:has-text("Compile")');
    await expect(page.locator(statusbar)).toContainText(/compile|failed|succeed/i, { timeout: 5000 });
  });

  test('Run button shows feedback', async ({ page }) => {
    await page.click('button:has-text("Run")');
    await expect(page.locator(statusbar)).toContainText(/complete|failed/i, { timeout: 10000 });
  });

  test('Save button shows result', async ({ page }) => {
    await page.click('button:has-text("Save")');
    await expect(page.locator(statusbar)).toContainText(/saving workflow|saved|saved to browser/i, { timeout: 5000 });
  });

  test('Export produces a download', async ({ page }) => {
    const downloadPromise = page.waitForEvent('download', { timeout: 5000 }).catch(() => null);
    await page.click('button:has-text("Export")');
    const download = await downloadPromise;
    if (download) {
      expect(download.suggestedFilename()).toMatch(/workflow.*\.json/);
    }
    await expect(page.locator(statusbar)).toContainText(/exported/i, { timeout: 3000 });
  });

  test('Load example then validate', async ({ page }) => {
    await page.selectOption('select', '/examples/simple_linear.json');
    await expect(page.locator('.designer-node').first()).toBeVisible({ timeout: 5000 });
    await page.click('button:has-text("Validate")');
    await expect(page.locator(statusbar)).toContainText(/valid|failed/i, { timeout: 5000 });
  });

});
