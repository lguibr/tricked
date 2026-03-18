import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
	plugins: [svelte({ hot: !process.env.VITEST })],
	test: {
		include: ['src/**/*.{test,spec}.{js,ts}'],
		environment: 'jsdom',
		globals: true,
		coverage: {
			exclude: [
				'src/routes/+layout.svelte',
				'src/app.d.ts',
				'src/lib/index.ts',
				'.svelte-kit/**',
				'svelte.config.js',
				'vite.config.ts',
				'vitest.config.ts',
				'**/*.test.ts'
			]
		}
	},
	resolve: {
		conditions: ['mode=test', 'browser'],
		alias: {
			$lib: '/Users/lg/lab/tricked/ui/src/lib'
		}
	}
});
