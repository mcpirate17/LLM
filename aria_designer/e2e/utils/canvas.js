import { expect } from '@playwright/test';

export async function waitForDesignerReady(page, timeout = 5000) {
  await expect(page.locator('.statusbar')).toBeVisible({ timeout });
  await expect(page.locator('.designer-node').first()).toBeVisible({ timeout });
}

export async function fitCanvasIfAvailable(page) {
  const fitButton = page.getByTitle('Fit to View');
  if (await fitButton.count()) {
    await fitButton.first().click();
  }
}

export async function clickFirstDesignerNode(page, timeout = 10000) {
  const nodes = page.locator('.designer-node');
  await expect(nodes.first()).toBeVisible({ timeout });
  try {
    await nodes.first().click({ force: true });
  } catch {
    await nodes.first().evaluate((el) => el.click());
  }
}
