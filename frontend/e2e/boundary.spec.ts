import { test, expect } from '@playwright/test';

test('App mounts without unhandled exceptions', async ({ page }) => {
  // Capture unhandled errors on the page
  const errors: Error[] = [];
  page.on('pageerror', (err) => {
    errors.push(err);
  });

  await page.goto('/');

  // Verify that there are no immediate unhandled errors
  expect(errors).toHaveLength(0);

  // Assert that some base UI elements are present
  // The control center should have a Sidebar or at least render text.
  // We'll wait for the body to be attached.
  await page.waitForSelector('body');

  // Verify that the title of the React App is Control Center or similar
  await expect(page).toHaveTitle(/Tricked/i);
});
