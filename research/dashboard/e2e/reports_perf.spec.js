const { test, expect } = require('@playwright/test');

const baseURL = process.env.E2E_BASE_URL;

test.describe('Reports performance', () => {
  test.beforeEach(async ({ page }) => {
    if (!baseURL) {
      test.skip(true, 'Set E2E_BASE_URL to run reports perf audit');
    }
    await page.goto(baseURL, { waitUntil: 'domcontentloaded' });
  });

  test('reports view opens within reasonable latency budgets', async ({ page }) => {
    await page.locator('button').filter({ hasText: /^Knowledge$/ }).first().click();

    let start = Date.now();
    await page.locator('button').filter({ hasText: /^Reports$/ }).first().click();
    await expect(page.getByText('Research Reports')).toBeVisible({ timeout: 30000 });
    const galleryMs = Date.now() - start;

    start = Date.now();
    await page.getByRole('button', { name: /All Time/i }).first().click();
    await expect(page.getByRole('button', { name: /Back to Reports/i })).toBeVisible({ timeout: 30000 });
    const detailHeaderMs = Date.now() - start;

    console.log(JSON.stringify({ galleryMs, detailHeaderMs }));

    expect(galleryMs).toBeLessThan(3000);
    expect(detailHeaderMs).toBeLessThan(3000);
  });
});
