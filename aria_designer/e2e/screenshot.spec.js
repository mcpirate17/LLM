import { test } from '@playwright/test';

test('screenshot: tropical block example', async ({ page }) => {
  await page.goto('/');
  await page.waitForTimeout(2000);
  await page.selectOption('.actions select', '/examples/tropical_block.json');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'screenshots/tropical_block.png', fullPage: false });
});

test('screenshot: transformer mini example', async ({ page }) => {
  await page.goto('/');
  await page.waitForTimeout(2000);
  await page.selectOption('.actions select', '/examples/transformer_mini.json');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'screenshots/transformer_mini.png', fullPage: false });
});

test('screenshot: starter workflow', async ({ page }) => {
  await page.goto('/');
  await page.waitForTimeout(2000);
  await page.screenshot({ path: 'screenshots/starter.png', fullPage: false });
});
