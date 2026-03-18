<script lang="ts">
	import { engine } from '$lib/state.svelte.ts';
	import { getPieceData } from '$lib/math';
</script>

<div class="grid grid-cols-3 gap-6 w-full max-w-3xl">
	{#each [0, 1, 2] as s}
		{@const p_id = engine.gameState ? engine.gameState.available[s] : -1}
		{@const isSelected = engine.selectedSlot === s}
		{@const pieceData = getPieceData(p_id, engine.gameState?.piece_masks)}

		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div
			class="bg-zinc-950/80 backdrop-blur-md border border-zinc-800 {isSelected
				? 'border-emerald-500 bg-emerald-500/5 scale-105 drop-shadow-[0_0_15px_rgba(16,185,129,0.15)]'
				: 'hover:border-zinc-500'} p-4 min-h-[160px] flex items-center justify-center flex-col relative cursor-pointer transition-all duration-200 rounded-none shadow-[0_0_30px_rgba(0,0,0,0.5)]"
			onclick={() => {
				if (!engine.isTraining && p_id !== -1) engine.selectedSlot = isSelected ? -1 : s;
			}}
			oncontextmenu={(e) => {
				e.preventDefault();
				engine.rotateSlot(s);
			}}
		>
			<span
				class="absolute top-3 left-3 text-[10px] font-bold uppercase tracking-widest text-zinc-600 font-mono"
				>Buffer {s}</span
			>
			{#if pieceData}
				<!-- Piece Render -->
				<svg
					class="w-full h-full drop-shadow-[0_0_8px_rgba(16,185,129,0.4)]"
					viewBox={pieceData.viewBox}
				>
					{#each pieceData.polys as pts}
						<polygon points={pts} class="fill-emerald-500 stroke-emerald-300/50 stroke-[1px]" />
					{/each}
				</svg>
			{/if}
		</div>
	{/each}
</div>

<div class="text-center mt-6 text-zinc-500 text-xs font-semibold tracking-widest uppercase">
	Right-Click to mathematically rotate buffer entities 60°
</div>
