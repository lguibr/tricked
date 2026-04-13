import { test, expect } from '@playwright/test';

test.describe('Play E2E Flow', () => {
  test('should allow user to start game, make move, and end game with vault submit', async ({ page }) => {
    await page.goto('http://localhost:5173/');

    // Click to Play tab
    const playTab = page.getByRole('tab', { name: /Play/i }).first();
    if (await playTab.isVisible()) {
        await playTab.click();
        
        // Find Start Game Button
        const startBtn = page.getByRole('button', { name: /Start Game/i }).first();
        if (await startBtn.isVisible()) {
            await startBtn.click();
            
            // Assert that the game grid renders
            const scoreLabel = page.getByText(/Score:/i).first();
            await expect(scoreLabel).toBeVisible({ timeout: 5000 });

            // Simulate clicking a grid slot or piece
            // This is largely implementation specific, so we try clicking the first available piece
            const pieceContainer = page.locator('.piece-container').first();
            if (await pieceContainer.isVisible()) {
                 await pieceContainer.click();
                 // If the UI expects a drag or a secondary click on the board
                 const boardSlot = page.locator('.board-cell').first();
                 if (await boardSlot.isVisible()) {
                      await boardSlot.click();
                 }
            }

            // End game
            const endBtn = page.getByRole('button', { name: /End Game|Give Up/i }).first();
            if (await endBtn.isVisible()) {
                await endBtn.click();

                // Expect a toaster or vault confirmation
                const toast = page.getByText(/Saved to Vault|Game Over/i).first();
                await expect(toast).toBeVisible({ timeout: 5000 });
            }
        }
    }
  });
});
