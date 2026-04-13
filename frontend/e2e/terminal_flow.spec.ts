import { test, expect } from '@playwright/test';

test.describe('Terminal E2E Flow', () => {
  test('should display log lines from a running process', async ({ page }) => {
    await page.goto('http://localhost:5173/');

    // Click on Dashboard tab where Terminal lives
    const dashboardTab = page.getByRole('tab', { name: /Dashboard/i }).first();
    if (await dashboardTab.isVisible()) {
        await dashboardTab.click();
    }
    
    // We expect the terminal element to exist. Usually inside a pre or code tag or custom container
    const terminalBlock = page.locator('pre, .terminal, [data-testid="terminal-container"]').first();
    await expect(terminalBlock).toBeVisible({ timeout: 5000 });

    // Since this is E2E, if a backend script is pumping out stdout, we should see it
    // Wait for some common output like "Tricked AI" or "Epoch" or just ANY non empty text content
    await expect(terminalBlock).not.toBeEmpty({ timeout: 5000 });
  });
});
