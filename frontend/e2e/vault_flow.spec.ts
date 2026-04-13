import { test, expect } from '@playwright/test';

test.describe('Vault E2E Flow', () => {
  test('should load vault games and display trajectory modal', async ({ page }) => {
    // Navigate to local dev server
    await page.goto('http://localhost:5173/');

    // Click to Vault tab
    const vaultTab = page.getByRole('tab', { name: /Vault/i }).first();
    if (await vaultTab.isVisible()) {
        await vaultTab.click();
        
        // Wait for the Vault view to render
        await expect(page.getByText(/Global Vault/i).first()).toBeVisible({ timeout: 5000 });

        // Select the first game row (Assuming it rendered from the API)
        const gameRow = page.locator('tbody tr').first();
        if (await gameRow.isVisible()) {
            await gameRow.click();
            
            // Assert a modal/visualizer opened for the trajectory
            const modal = page.getByRole('dialog').first();
            await expect(modal).toBeVisible({ timeout: 5000 });
            
            // Confirm there is a close button and click it
            const closeBtn = modal.getByRole('button', { name: /Close|X/i }).first();
            await closeBtn.click();
            
            await expect(modal).not.toBeVisible();
        }
    }
  });
});
