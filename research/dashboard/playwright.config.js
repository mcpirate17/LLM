// @ts-check
const { defineConfig } = require('@playwright/test');

const baseURL = process.env.E2E_BASE_URL || '';

module.exports = defineConfig({
  testDir: './e2e',
  timeout: 120000,
  expect: { timeout: 15000 },
  use: {
    baseURL,
    headless: true,
    viewport: { width: 1440, height: 900 },
    trace: 'retain-on-failure',
  },
  reporter: [['list']],
});
