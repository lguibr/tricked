import { test, expect } from '@playwright/test';

test.describe('System Telemetry E2E Flow', () => {
  test('should display hardware topography and active percentages', async ({ page }) => {
    await page.goto('http://localhost:5173/');

    // Click on Dashboard tab where Hardware telemetry lives
    const dashboardTab = page.getByRole('tab', { name: /Dashboard/i }).first();
    if (await dashboardTab.isVisible()) {
        await dashboardTab.click();
    }
    
    // Hardware widget displays CPU & RAM indicators
    const cpuHeader = page.getByText(/CPU/i).first();
    await expect(cpuHeader).toBeVisible({ timeout: 5000 });

    const ramHeader = page.getByText(/RAM/i).first();
    await expect(ramHeader).toBeVisible({ timeout: 5000 });
    
    // It should render real values, e.g. "XX%"
    await expect(page.getByText(/%/i).first()).toBeVisible({ timeout: 5000 });
  });
});
