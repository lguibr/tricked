<script lang="ts">
	import '../app.css';
	import { onMount, onDestroy } from 'svelte';
	import { engine } from '$lib/state.svelte';

	onMount(() => {
		engine.mount();
	});

	onDestroy(() => {
		engine.unmount();
	});
</script>

<div
	class="bg-surface-container-lowest text-on-surface font-body selection:bg-primary-container selection:text-on-primary-fixed overflow-hidden flex flex-col h-screen"
>
	<main class="flex-1 relative overflow-y-auto">
		<slot />
	</main>
</div>
