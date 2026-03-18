<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { engine } from '$lib/state.svelte.ts';
	import MatrixBackground from '$lib/components/MatrixBackground.svelte';
	import ControlsBar from '$lib/components/ControlsBar.svelte';
	import TriangoBoard from '$lib/components/TriangoBoard.svelte';
	import VaultDataGrid from '$lib/components/VaultDataGrid.svelte';
	import EpochTelemetry from '$lib/components/EpochTelemetry.svelte';
	import PieceBuffer from '$lib/components/PieceBuffer.svelte';

	onMount(() => {
		engine.mount();
	});

	onDestroy(() => {
		engine.unmount();
	});
</script>

<svelte:head>
	<title>Tricked: AI Engine</title>
</svelte:head>

<main class="min-h-screen flex flex-col items-center justify-center p-8 font-sans">
	{#if engine.loading}
		<div class="text-emerald-500 animate-pulse text-2xl font-bold">
			Booting Svelte Neural Interface...
		</div>
	{:else}
		<MatrixBackground topGames={engine.topGames} />

		<!-- Header -->
		<div class="mb-6 text-center z-10">
			<h1
				class="text-5xl font-black tracking-tight text-transparent bg-clip-text bg-gradient-to-br from-emerald-400 to-cyan-500 mb-2 drop-shadow-sm"
			>
				Tricked
			</h1>
			<p class="text-zinc-500 font-medium tracking-wide">120-Degree Mathematical Engine</p>
		</div>

		<ControlsBar />
		<TriangoBoard />
		<VaultDataGrid />

		<!-- Footer Stats -->
		<div class="w-full max-w-3xl flex justify-between items-end mb-6 px-4">
			<div class="text-3xl font-bold tracking-tight text-zinc-200">
				Efficiency: <span class="text-emerald-400">{engine.gameState?.score || 0}</span>
			</div>
			<div class="text-zinc-500 font-medium">
				Entities Remaining: <span class="text-zinc-300">{engine.gameState?.pieces_left || 0}</span>
			</div>
		</div>

		<PieceBuffer />
	{/if}

	<EpochTelemetry />
</main>
