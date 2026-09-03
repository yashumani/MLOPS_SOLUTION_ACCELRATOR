import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  testMatch: "**/*.pw.ts",
  workers: 1,
  retries: 0,
  timeout: 30_000,
  use: { baseURL: "http://127.0.0.1:8510", browserName: "chromium", channel: process.env.PLAYWRIGHT_CHANNEL || undefined, headless: true },
  webServer: { command: "npm run dev -- --host 127.0.0.1 --port 8510 --strictPort", url: "http://127.0.0.1:8510", reuseExistingServer: !process.env.CI, timeout: 30_000 }
});
