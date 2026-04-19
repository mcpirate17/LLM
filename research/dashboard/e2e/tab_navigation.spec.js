const { test, expect } = require('@playwright/test');

const baseURL = process.env.E2E_BASE_URL;

function attachErrorCollectors(page) {
  const pageErrors = [];
  const requestFailures = [];
  page.on('pageerror', err => pageErrors.push(String(err)));
  page.on('console', msg => {
    if (msg.type() === 'error') pageErrors.push(msg.text());
  });
  page.on('response', async (response) => {
    const status = response.status();
    if (status >= 500) {
      requestFailures.push(`${status} ${response.url()}`);
    }
  });
  page.on('requestfailed', request => {
    requestFailures.push(`FAILED ${request.method()} ${request.url()} :: ${request.failure()?.errorText || 'unknown'}`);
  });
  return { pageErrors, requestFailures };
}

function meaningfulPageErrors(errors) {
  return errors.filter((msg) => !msg.includes('status of 404'));
}

function meaningfulRequestFailures(failures) {
  return failures.filter((msg) => !msg.includes('/api/observability/stream') && !msg.includes('ERR_ABORTED'));
}

test.describe('Dashboard tab navigation', () => {
  test.beforeEach(async ({ page }) => {
    if (!baseURL) {
      test.skip(true, 'Set E2E_BASE_URL to run dashboard navigation audit');
    }
    await page.goto(baseURL, { waitUntil: 'domcontentloaded' });
    await expect(page.getByRole('button', { name: /^Refresh$/ }).first()).toBeVisible({ timeout: 30000 });
  });

  test('top-level tabs render without JS or 5xx errors', async ({ page }) => {
    const { pageErrors, requestFailures } = attachErrorCollectors(page);

    const steps = [
      { primary: 'Workbench', label: 'Command' },
      { primary: 'Workbench', label: 'Experiments' },
      { primary: 'Workbench', label: 'Discoveries' },
      { primary: 'Knowledge', label: 'Analytics' },
      { primary: 'Knowledge', label: 'Reports' },
      { primary: 'Knowledge', label: 'Decisions' },
      { primary: 'Knowledge', label: 'Log' },
      { primary: 'Diagnostics', label: 'Template & Slots' },
      { primary: 'Diagnostics', label: 'Components' },
      { primary: 'Diagnostics', label: 'Infrastructure' },
      { primary: 'Diagnostics', label: 'Optimization' },
      { primary: 'Diagnostics', label: 'References' },
    ];

    for (const step of steps) {
      await page.getByRole('button', { name: new RegExp(`^${step.primary}$`) }).first().click();
      const tabButton = page.getByRole('button', { name: new RegExp(`^${step.label}$`) }).first();
      await tabButton.click();
      await expect(tabButton).toHaveClass(/active/, { timeout: 30000 });
    }

    expect(meaningfulPageErrors(pageErrors)).toEqual([]);
    expect(meaningfulRequestFailures(requestFailures)).toEqual([]);
  });

  test('template tab triggers a fresh full dashboard fetch when opened', async ({ page }) => {
    const { pageErrors, requestFailures } = attachErrorCollectors(page);

    await page.getByRole('button', { name: /^Workbench$/ }).first().click();
    await page.getByRole('button', { name: /^Experiments$/ }).first().click();
    await expect(page.getByText('Experiments').first()).toBeVisible({ timeout: 30000 });

    const dashboardResponses = [];
    page.on('response', async (response) => {
      const url = response.url();
      if (url.includes('/api/dashboard')) {
        dashboardResponses.push(url);
      }
    });

    await Promise.all([
      page.waitForResponse((response) => {
        const url = response.url();
        return url.includes('/api/dashboard') && response.status() === 200;
      }, { timeout: 30000 }),
      page.getByRole('button', { name: /^Diagnostics$/ }).first().click(),
      page.getByRole('button', { name: /^Template & Slots$/ }).first().click(),
    ]);

    await expect(page.getByText('Template & Slot Observability', { exact: false }).first()).toBeVisible({ timeout: 30000 });
    expect(dashboardResponses.some(url => url.includes('/api/dashboard'))).toBeTruthy();
    expect(meaningfulPageErrors(pageErrors)).toEqual([]);
    expect(meaningfulRequestFailures(requestFailures)).toEqual([]);
  });

  test('reports sub-actions and global refresh stay usable after tab switches', async ({ page }) => {
    const { pageErrors, requestFailures } = attachErrorCollectors(page);

    await page.getByRole('button', { name: /^Knowledge$/ }).first().click();
    await page.getByRole('button', { name: /^Reports$/ }).first().click();
    await expect(page.getByText('Research Reports')).toBeVisible({ timeout: 30000 });

    await page.getByRole('button', { name: /Load Campaigns/i }).click();
    await expect(page.getByText('Research Campaigns')).toBeVisible({ timeout: 30000 });

    await page.getByRole('button', { name: /Load Knowledge Base/i }).click();
    await expect(page.locator('h2, h3').filter({ hasText: /^Knowledge Base$/ }).first()).toBeVisible({ timeout: 30000 });

    await page.getByRole('button', { name: /^Refresh$/ }).first().click();
    await expect(page.getByText('Research Reports')).toBeVisible({ timeout: 30000 });

    expect(meaningfulPageErrors(pageErrors)).toEqual([]);
    expect(meaningfulRequestFailures(requestFailures)).toEqual([]);
  });
});
