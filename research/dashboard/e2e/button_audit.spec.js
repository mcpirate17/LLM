const { test, expect } = require('@playwright/test');

const baseURL = process.env.E2E_BASE_URL;

test.describe('UI action audit', () => {
  test.beforeEach(async ({ page }) => {
    if (!baseURL) {
      test.skip(true, 'Set E2E_BASE_URL to run UI action audit');
    }
    await page.goto(baseURL, { waitUntil: 'domcontentloaded' });
  });

  test('click all visible buttons and links', async ({ page }) => {
    const consoleErrors = [];
    page.on('pageerror', err => consoleErrors.push(String(err)));
    page.on('console', msg => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });

    const getCandidates = async () => (
      page.$$('button, a[role="button"], a[href], [role="button"]')
    );

    const shouldSkip = async (handle) => {
      const isVisible = await handle.isVisible().catch(() => false);
      if (!isVisible) return true;
      const disabled = await handle.getAttribute('disabled');
      if (disabled !== null) return true;
      const ariaDisabled = await handle.getAttribute('aria-disabled');
      if (ariaDisabled === 'true') return true;
      return false;
    };

    let clicked = 0;
    const seen = new Set();
    const candidates = await getCandidates();

    for (const handle of candidates) {
      if (await shouldSkip(handle)) continue;
      const key = await handle.evaluate(el => {
        const txt = (el.innerText || '').trim();
        const id = el.getAttribute('id') || '';
        const role = el.getAttribute('role') || el.tagName;
        return `${role}:${id}:${txt}`.slice(0, 120);
      });
      if (seen.has(key)) continue;
      seen.add(key);

      await handle.scrollIntoViewIfNeeded().catch(() => {});
      await handle.click({ timeout: 2000 }).catch(() => {});
      clicked += 1;
    }

    expect(clicked).toBeGreaterThan(0);
    expect(consoleErrors.length).toBe(0);
  });

  test('exercise form fields and table controls', async ({ page }) => {
    const consoleErrors = [];
    page.on('pageerror', err => consoleErrors.push(String(err)));
    page.on('console', msg => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });

    const typeInto = async (selector, value) => {
      const handles = await page.$$(selector);
      for (const handle of handles) {
        const visible = await handle.isVisible().catch(() => false);
        if (!visible) continue;
        const disabled = await handle.getAttribute('disabled');
        if (disabled !== null) continue;
        await handle.click({ timeout: 2000 }).catch(() => {});
        await handle.fill(String(value)).catch(() => {});
      }
    };

    await typeInto('input[type="text"], input:not([type])', 'e2e');
    await typeInto('input[type="search"]', 'e2e');
    await typeInto('input[type="number"]', '1');
    await typeInto('textarea', 'e2e');

    const selects = await page.$$('select');
    for (const select of selects) {
      const visible = await select.isVisible().catch(() => false);
      if (!visible) continue;
      const opts = await select.$$('option');
      if (opts.length > 1) {
        const value = await opts[1].getAttribute('value');
        if (value !== null) {
          await select.selectOption(value).catch(() => {});
        }
      }
    }

    const checkboxes = await page.$$('input[type="checkbox"], input[type="radio"]');
    for (const box of checkboxes) {
      const visible = await box.isVisible().catch(() => false);
      if (!visible) continue;
      const disabled = await box.getAttribute('disabled');
      if (disabled !== null) continue;
      await box.click({ timeout: 2000 }).catch(() => {});
    }

    // Table headers (potential sortable columns)
    const headers = await page.$$('table thead th');
    for (const header of headers) {
      const visible = await header.isVisible().catch(() => false);
      if (!visible) continue;
      await header.click({ timeout: 2000 }).catch(() => {});
    }

    // Generic role-based table headers
    const roleHeaders = await page.$$('[role="columnheader"]');
    for (const header of roleHeaders) {
      const visible = await header.isVisible().catch(() => false);
      if (!visible) continue;
      await header.click({ timeout: 2000 }).catch(() => {});
    }

    expect(consoleErrors.length).toBe(0);
  });

  test('open and dismiss dialogs/menus when present', async ({ page }) => {
    const consoleErrors = [];
    page.on('pageerror', err => consoleErrors.push(String(err)));
    page.on('console', msg => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });

    const triggers = await page.$$(
      '[aria-haspopup="dialog"], [aria-haspopup="menu"], [data-modal], [data-menu]'
    );
    for (const trigger of triggers) {
      const visible = await trigger.isVisible().catch(() => false);
      if (!visible) continue;
      await trigger.scrollIntoViewIfNeeded().catch(() => {});
      await trigger.click({ timeout: 2000 }).catch(() => {});
      const dialogs = await page.$$('dialog, [role="dialog"], [aria-modal="true"]');
      for (const dlg of dialogs) {
        const dlgVisible = await dlg.isVisible().catch(() => false);
        if (dlgVisible) {
          // Try close buttons inside dialog
          const closeBtn = await dlg.$('button[aria-label="Close"], button[title="Close"], button:has-text("Close")');
          if (closeBtn) {
            await closeBtn.click({ timeout: 2000 }).catch(() => {});
          } else {
            await page.keyboard.press('Escape').catch(() => {});
          }
        }
      }
    }

    expect(consoleErrors.length).toBe(0);
  });
});
