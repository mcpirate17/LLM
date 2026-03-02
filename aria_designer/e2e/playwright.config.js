import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: '.',
  timeout: 30_000,
  retries: 0,
  use: {
    baseURL: 'http://localhost:5176',
    headless: true,
    screenshot: 'only-on-failure',
  },
  webServer: [
    {
      command: 'bash -lc "source /home/tim/venvs/llm/bin/activate && cd ../api && python -m uvicorn app.main:app --port 8091"',
      port: 8091,
      timeout: 60_000,
      reuseExistingServer: true,
    },
    {
      command: 'cd ../ui && npm run dev -- --port 5176 --strictPort',
      port: 5176,
      timeout: 60_000,
      reuseExistingServer: true,
    },
  ],
});
