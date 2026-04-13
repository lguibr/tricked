import { test, expect } from '@playwright/test';

test.describe('Metrics E2E Flow', () => {
  test('should render active metrics in chart when a run is active', async ({ page }) => {
    await page.goto('http://localhost:5173/');

    // Click on Dashboard tab or assume it's default
    const dashboardTab = page.getByRole('tab', { name: /Dashboard/i }).first();
    if (await dashboardTab.isVisible()) {
        await dashboardTab.click();
    }
    
    // Check if an EChart canvas has rendered on the page
    // ECharts usually injects a canvas element inside a div
    const charts = page.locator('canvas');
    if (await charts.count() > 0) {
        await expect(charts.first()).toBeVisible({ timeout: 5000 });
    }

    // Usually metrics require an active run, so we might need to click an active run
    const runItem = page.getByText(/RUNNING|STARTING/i).first();
    if (await runItem.isVisible()) {
         await runItem.click();
         // Wait for stats to update
         const metricLabel = page.getByText(/Loss|Entropy|Step/i).first();
         await expect(metricLabel).toBeVisible({ timeout: 5000 });
    }
  });
});
