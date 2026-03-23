<script lang="ts">
	import { engine } from '$lib/state.svelte';
	import BackgroundHexGrid from '$lib/components/BackgroundHexGrid.svelte';
	import ExperimentOrchestrator from '$lib/components/ExperimentOrchestrator.svelte';
	import TriangoBoard from '$lib/components/TriangoBoard.svelte';
	import PieceBuffer from '$lib/components/PieceBuffer.svelte';
</script>

<div
	class="min-h-screen relative w-full overflow-hidden flex flex-col md:flex-row items-center justify-center p-4 gap-8"
>
	
	<BackgroundHexGrid topGames={engine.topGames.slice(0, 32)} />

	<div class="relative z-10 w-full max-w-xl flex flex-col items-center gap-4">
		{#if engine.gameState}
			<TriangoBoard />
			<PieceBuffer />
		{:else}
			<div
				class="w-full aspect-square border border-dashed border-outline-variant/30 flex items-center justify-center text-on-surface-variant font-mono text-sm opacity-50 backdrop-blur-md bg-surface/20"
			>
				Awaiting Telemetry...
			</div>
		{/if}
	</div>

	<div class="relative z-10 w-full max-w-2xl">
		<ExperimentOrchestrator />
	</div>
</div>
