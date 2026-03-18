import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/svelte';
import Page from './+page.svelte';

describe('Tricked +page.svelte', () => {
	it('mounts the application shell correctly in loading state', () => {
		const { getByText } = render(Page);
		expect(getByText(/Booting Svelte Neural Interface/i)).toBeDefined();
	});
});
