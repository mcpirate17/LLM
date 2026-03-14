import { test, expect } from '@playwright/test';
import { waitForDesignerReady } from './utils/canvas.js';

test.describe('Aria Chat', () => {
  test('can open chat and send a message', async ({ page }) => {
    await page.goto('/');
    await waitForDesignerReady(page);

    // Click on "Aria Chat" tab
    const chatTab = page.locator('button:has-text("Aria Chat")');
    await expect(chatTab).toBeVisible();
    await chatTab.click();

    // Verify chat panel is visible
    const chatPanel = page.locator('.chat-panel');
    await expect(chatPanel).toBeVisible();

    // Type a message
    const input = chatPanel.locator('input[placeholder="Describe what you want to build..."]');
    await input.fill('Build a simple linear model');
    
    // Click send
    const sendBtn = chatPanel.locator('button:has-text("Send")');
    await sendBtn.click();

    // Verify message appears in chat
    await expect(chatPanel.locator('.chat-bubble.user')).toContainText('Build a simple linear model');
    
    // Verify Aria response (might take a moment)
    await expect(chatPanel.locator('.chat-bubble.aria')).toBeVisible({ timeout: 10000 });
  });

  test('can switch to chat from Ask Aria modal', async ({ page }) => {
    await page.goto('/');
    await waitForDesignerReady(page);

    // Click "Ask Aria" in header
    const askAriaBtn = page.locator('button:has-text("Ask Aria")');
    await askAriaBtn.click();

    // Modal should be visible
    const modal = page.locator('.modal-overlay');
    await expect(modal).toBeVisible();

    // Click "Switch to Chat"
    const switchToChatBtn = modal.locator('button:has-text("Switch to Chat")');
    await switchToChatBtn.click();

    // Modal should close and chat tab should be active
    await expect(modal).not.toBeVisible();
    const chatTab = page.locator('button.active:has-text("Aria Chat")');
    await expect(chatTab).toBeVisible();
  });

  test('can save and see fingerprint link', async ({ page }) => {
    await page.goto('/');
    await waitForDesignerReady(page);

    // Click "File" -> "Save"
    await page.locator('button:has-text("File")').click();
    const saveBtn = page.locator('button:has-text("Save")');
    await saveBtn.click();

    // Verify status message or save feedback
    await expect(page.locator('.save-feedback.save-saved')).toBeVisible({ timeout: 10000 });
    
    // Verify fingerprint link is visible
    const fpLink = page.locator('.fingerprint-link');
    await expect(fpLink).toBeVisible();
    await expect(fpLink).toHaveAttribute('target', '_blank');
    await expect(fpLink).toHaveAttribute('href', /localhost:5000/);
  });
});
