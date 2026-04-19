const { test, expect } = require('@playwright/test');

const baseURL = process.env.E2E_BASE_URL;

function meaningfulErrors(errors) {
  return errors.filter((msg) => {
    const text = String(msg || '');
    if (text.includes('404')) return false;
    if (text.includes('/api/observability/stream')) return false;
    return true;
  });
}

test.describe('Command and diagnostics surfaces', () => {
  test.beforeEach(async ({ page }) => {
    if (!baseURL) {
      test.skip(true, 'Set E2E_BASE_URL to run command/diagnostics audit');
    }
    await page.goto(baseURL, { waitUntil: 'domcontentloaded' });
  });

  test('workbench shows idle state instead of stale running state', async ({ page }) => {
    await page.locator('button').filter({ hasText: /^Workbench$/ }).first().click();
    await expect(page.getByText(/Idle — ready for next run|Idle —/i).first()).toBeVisible({ timeout: 30000 });
    await expect(page.getByText(/Running experiment/i)).toHaveCount(0);
    await expect(page.getByText(/Autonomous cycle .* Running/i)).toHaveCount(0);
  });

  test('status-bar diagnostics toggle and analytics diagnostics render without errors', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (err) => errors.push(String(err)));
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });

    await page.locator('button').filter({ hasText: /^Workbench$/ }).first().click();
    const diagnosticsToggle = page.getByRole('button', { name: /Show technical diagnostics|Hide technical diagnostics/i }).first();
    await expect(diagnosticsToggle).toBeVisible({ timeout: 30000 });
    await diagnosticsToggle.click();
    await expect(page.getByRole('button', { name: /Hide technical diagnostics/i })).toBeVisible({ timeout: 30000 });
    await page.getByRole('button', { name: /Hide technical diagnostics/i }).click();
    await expect(page.getByRole('button', { name: /Show technical diagnostics/i })).toBeVisible({ timeout: 30000 });

    await page.locator('button').filter({ hasText: /^Knowledge$/ }).first().click();
    await page.locator('button').filter({ hasText: /^Analytics$/ }).first().click();
    await expect(page.getByText('Research Trends').first()).toBeVisible({ timeout: 30000 });

    await page.getByRole('button', { name: /^Learning$/ }).first().click();
    await expect(page.getByText('Core Learning')).toBeVisible({ timeout: 30000 });
    await page.getByText('Advanced Diagnostics').click();
    await expect(page.getByText('Fingerprint Diagnostics')).toBeVisible({ timeout: 30000 });
    await expect(page.getByText(/Sensitivity skips:/i)).toBeVisible({ timeout: 30000 });

    expect(meaningfulErrors(errors)).toEqual([]);
  });
});
