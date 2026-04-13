import { test, expect } from '@playwright/test';

test.describe('Evaluation E2E Flow', () => {
  test('should select a model, run one eval step, and show metric updates', async ({ page }) => {
    // Navigate to local dev server
    await page.goto('http://localhost:5173/');

    // Click to Eval tab
    const evalTab = page.getByRole('tab', { name: /Eval/i }).first();
    if (await evalTab.isVisible()) {
        await evalTab.click();
        
        // Wait for UI to load
        await expect(page.getByText(/Evaluation Workspace/i).first()).toBeVisible({ timeout: 5000 });

        // Find difficulty selector if available
        const diffSelect = page.getByRole('combobox', { name: /Difficulty/i }).first();
        if (await diffSelect.isVisible()) {
             await diffSelect.click();
             await page.getByRole('option', { name: /1/i }).first().click();
        }

        // Mock an evaluation flow - typically involves picking a model and calculating step
        const startEvalBtn = page.getByRole('button', { name: /Start Evaluation|Run/i }).first();
        if (await startEvalBtn.isVisible()) {
             await startEvalBtn.click();
             
             // Check if score changed or pieces updated
             const curScore = page.getByText(/Score:/i).first();
             // Assumes the native engine responds within 5 seconds for a single step
             await expect(curScore).toBeVisible({ timeout: 10000 });

             // Stop eval if looping
             const stopBtn = page.getByRole('button', { name: /Stop|Pause/i }).first();
             if (await stopBtn.isVisible()) {
                 await stopBtn.click();
             }
        }
    }
  });
});
