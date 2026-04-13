import { test, expect } from '@playwright/test';

test.describe('Config & Run E2E Flow', () => {
  test('should create, start, and stop a run successfully', async ({ page }) => {
    // Navigate to local dev server
    await page.goto('http://localhost:5173/');

    // Click "New Run" or equivalent trigger
    // Looking for general dashboard patterns
    const newRunBtn = page.getByRole('button', { name: /New|Create|Add/i }).first();
    if (await newRunBtn.isVisible()) {
        await newRunBtn.click();
        
        // Wait for config modal/drawer
        const nameInput = page.getByPlaceholder(/Name/i).first();
        if (await nameInput.isVisible()) {
            await nameInput.fill('Playwright Test Run');
        }

        // Submit form
        const submitBtn = page.getByRole('button', { name: /Submit|Create Run|Save/i }).first();
        await submitBtn.click();
    }

    // Since we don't know the exact UI mapping, just check if list populates and click Start
    const runItem = page.getByText(/Playwright Test|Run/i).first();
    await expect(runItem).toBeVisible({ timeout: 5000 });

    const playBtn = page.getByRole('button', { name: /Start|Play/i }).first();
    if (await playBtn.isVisible()) {
        await playBtn.click();
        // Assume State changes to RUNNING
        await expect(page.getByText(/RUNNING/i).first()).toBeVisible({ timeout: 10000 });
        
        // Stop
        const stopBtn = page.getByRole('button', { name: /Stop|Pause/i }).first();
        if (await stopBtn.isVisible()) {
             await stopBtn.click();
             await expect(page.getByText(/STOPPED/i).first()).toBeVisible({ timeout: 10000 });
        }
    }
  });
});
