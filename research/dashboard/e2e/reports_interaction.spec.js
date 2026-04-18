const { test, expect } = require('@playwright/test');

const baseURL = process.env.E2E_BASE_URL;

test.describe('Reports interaction', () => {
  test.beforeEach(async ({ page }) => {
    if (!baseURL) {
      test.skip(true, 'Set E2E_BASE_URL to run reports interaction audit');
    }
    await page.goto(baseURL, { waitUntil: 'domcontentloaded' });
  });

  test('report gallery stays clickable and other tabs remain usable', async ({ page }) => {
    await page.locator('button').filter({ hasText: /^Knowledge$/ }).first().click();
    await page.locator('button').filter({ hasText: /^Reports$/ }).first().click();
    await expect(page.getByText('Research Reports')).toBeVisible({ timeout: 30000 });
    await expect(page.getByRole('button', { name: /Load Campaigns/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /Load Knowledge Base/i })).toBeVisible();

    const allTimeCard = page.getByRole('button', { name: /All Time/i }).first();
    await allTimeCard.click();
    await expect(page.getByRole('button', { name: /Back to Reports/i })).toBeVisible({ timeout: 30000 });

    const thisWeekCard = page.getByRole('button', { name: /This Week/i }).first();
    await expect(thisWeekCard).toBeVisible();
    await thisWeekCard.click();
    await expect(page.getByText('This Week', { exact: true })).toBeVisible({ timeout: 30000 });

    await page.locator('button').filter({ hasText: /^Analytics$/ }).first().click();
    await expect(page.locator('button').filter({ hasText: /^Reports$/ }).first()).toBeVisible();
  });

  test('campaign and knowledge sections load on demand', async ({ page }) => {
    await page.locator('button').filter({ hasText: /^Knowledge$/ }).first().click();
    await page.locator('button').filter({ hasText: /^Reports$/ }).first().click();
    await expect(page.getByText('Research Reports')).toBeVisible({ timeout: 30000 });

    await page.getByRole('button', { name: /Load Campaigns/i }).click();
    await expect(page.getByText('Research Campaigns')).toBeVisible({ timeout: 30000 });

    await page.getByRole('button', { name: /Load Knowledge Base/i }).click();
    await expect(page.getByText('Knowledge Base')).toBeVisible({ timeout: 30000 });
  });
});
