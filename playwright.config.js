import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "tests",
  testMatch: "**/*.spec.js",
  timeout: 60000,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:4178",
    channel: "chrome",
    viewport: { width: 1280, height: 800 },
  },
  webServer: {
    command: "npm run start",
    url: "http://127.0.0.1:4178",
    reuseExistingServer: false,
    timeout: 30000,
  },
});
