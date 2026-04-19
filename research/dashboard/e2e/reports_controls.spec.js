const { test, expect } = require('@playwright/test');

const baseURL = process.env.E2E_BASE_URL;

function meaningfulPageErrors(errors) {
  return errors.filter((msg) => {
    const text = String(msg || '');
    return !text.includes('404');
  });
}

test.describe('Reports controls', () => {
  test.beforeEach(async ({ page }) => {
    if (!baseURL) {
      test.skip(true, 'Set E2E_BASE_URL to run reports control audit');
    }
    await page.goto(baseURL, { waitUntil: 'domcontentloaded' });
    await page.locator('button').filter({ hasText: /^Knowledge$/ }).first().click();
    await page.locator('button').filter({ hasText: /^Reports$/ }).first().click();
    await expect(page.getByText('Research Reports')).toBeVisible({ timeout: 30000 });
  });

  test('scoped report controls work without page errors', async ({ page }) => {
    const consoleErrors = [];
    page.on('pageerror', (err) => consoleErrors.push(String(err)));
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    await page.getByRole('button', { name: /Compression/i }).first().click();
    await expect(page.getByText('Compression Report')).toBeVisible({ timeout: 30000 });
    await expect(page.getByRole('button', { name: /Back to Reports/i })).toBeVisible();

    const themeSelect = page.locator('select').nth(0);
    const trendSelect = page.locator('select').nth(1);
    const topKInput = page.locator('input[type="number"]').first();

    await themeSelect.selectOption('routing');
    await trendSelect.selectOption('high_survival');
    await topKInput.fill('7');
    await page.getByRole('button', { name: /Generate Scoped Report/i }).click();

    await expect(page.getByText(/Query: theme=routing/i)).toBeVisible({ timeout: 30000 });
    await expect(page.getByText(/trend=high_survival/i)).toBeVisible({ timeout: 30000 });

    await page.getByRole('button', { name: /Reset to Fast Overview/i }).click();
    await expect(page.getByText(/Source: \/api\/report\/query/i)).toBeVisible({ timeout: 30000 });

    await page.getByRole('button', { name: /Load Full Details/i }).click();
    await expect(page.getByText(/Source: \/api\/report/i)).toBeVisible({ timeout: 30000 });

    expect(meaningfulPageErrors(consoleErrors)).toEqual([]);
  });

  test('markdown export triggers a download', async ({ page }) => {
    await page.getByRole('button', { name: /All Time/i }).first().click();
    await expect(page.getByText('All Time Report')).toBeVisible({ timeout: 30000 });

    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: /Export Markdown/i }).click();
    const download = await downloadPromise;

    expect(download.suggestedFilename()).toMatch(/^research_report_\d{4}-\d{2}-\d{2}\.md$/);
  });
});
