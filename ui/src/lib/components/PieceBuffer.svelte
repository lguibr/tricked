<script lang="ts">
	import { engine } from '$lib/state.svelte.js';
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
			class="bg-surface/80 backdrop-blur-md border border-outline-variant/20 {isSelected
				? 'border-primary bg-primary/5 scale-105 drop-shadow-[0_0_40px_var(--color-surface-tint)]'
				: 'hover:border-primary/50'} p-4 min-h-[160px] flex items-center justify-center flex-col relative cursor-pointer transition-all duration-200 rounded-none z-10"
			onclick={() => {
				if (!engine.isTraining && p_id !== -1) engine.selectedSlot = isSelected ? -1 : s;
			}}
			oncontextmenu={(e) => {
				e.preventDefault();
				engine.rotateSlot(s);
			}}
		>
			<span
				class="absolute top-3 left-3 text-[10px] font-bold uppercase tracking-widest text-on-surface-variant font-mono"
				>Buffer {s}</span
			>
			{#if pieceData}
				<!-- Piece Render -->
				<svg
					class="w-full h-full drop-shadow-[0_0_40px_var(--color-surface-tint)]"
					viewBox={pieceData.viewBox}
				>
					{#each pieceData.polys as pts}
						<polygon points={pts} class="fill-primary stroke-primary-dim/50 stroke-[1px]" />
					{/each}
				</svg>
			{/if}
		</div>
	{/each}
</div>

<div class="text-center mt-6 text-on-surface-variant text-[10px] font-semibold tracking-widest uppercase">
	Right-Click to mathematically rotate buffer entities 60°
</div>
