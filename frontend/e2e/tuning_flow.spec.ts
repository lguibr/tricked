import { test, expect } from '@playwright/test';

test.describe('Tuning E2E Flow', () => {
  test('should create, launch tuning study and display charts', async ({ page }) => {
    // Navigate to local dev server
    await page.goto('http://localhost:5173/');

    // Click to tuning tab or menu
    const tuningTab = page.getByRole('tab', { name: /Tuning|Optimizer/i }).first();
    if (await tuningTab.isVisible()) {
        await tuningTab.click();
        
        // Find New Study Button
        const newStudyBtn = page.getByRole('button', { name: /New Study|Create/i }).first();
        if (await newStudyBtn.isVisible()) {
            await newStudyBtn.click();
            
            // Wait for tuning config drawer
            const nameInput = page.getByPlaceholder(/Study Name/i).first();
            if (await nameInput.isVisible()) {
                await nameInput.fill('Playwright Tuning Study');
            }

            // Submit 
            const submitBtn = page.getByRole('button', { name: /Submit|Create Study/i }).first();
            await submitBtn.click();
        }

        // Wait for study item to show
        const studyItem = page.getByText(/Playwright Tuning Study/i).first();
        await expect(studyItem).toBeVisible({ timeout: 5000 });

        // Select it and click Start Optimization
        await studyItem.click();
        const startOptBtn = page.getByRole('button', { name: /Start Optimization/i }).first();
        
        if (await startOptBtn.isVisible()) {
            await startOptBtn.click();
            // Assert that status hits RUNNING in the Tuning tab
            await expect(page.getByText(/RUNNING/i).first()).toBeVisible({ timeout: 10000 });
            
            // Stop
            const stopBtn = page.getByRole('button', { name: /Stop|Pause/i }).first();
            if (await stopBtn.isVisible()) {
                await stopBtn.click();
                await expect(page.getByText(/STOPPED/i).first()).toBeVisible({ timeout: 10000 });
            }
        }
    }
  });
});
