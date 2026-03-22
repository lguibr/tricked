<script lang="ts">
	import { engine } from '$lib/state.svelte.js';
</script>

{#if engine.isTraining && engine.trainingInfo}
	<div
		class="fixed top-6 right-6 z-50 bg-surface/90 backdrop-blur-xl border border-outline-variant/20 drop-shadow-[0_0_40px_var(--color-surface-tint)] p-4 rounded-none flex flex-col gap-2 min-w-[280px]"
	>
		<div class="flex items-center justify-between mb-1">
			<span
				class="text-[10px] font-black tracking-widest uppercase text-primary animate-pulse flex items-center gap-2"
				><span class="w-2 h-2 rounded-none bg-primary"></span> Training Active</span
			>
		</div>
		{#if engine.trainingInfo.iteration}
			<div class="flex flex-col gap-1 items-start text-left w-full">
				<span class="text-xs font-mono font-bold text-on-surface w-full mb-2"
					>{engine.trainingInfo.stage || 'Initializing...'}</span
				>
				<div
					class="flex justify-between w-full text-[10px] text-on-surface-variant font-bold uppercase tracking-widest"
				>
					<span>Epoch {engine.trainingInfo.iteration}/{engine.trainingInfo.total_iterations}</span>
					<span class="text-secondary"
						>Med: {Math.round(engine.trainingInfo.median_score || 0)} / Max: {Math.round(
							engine.trainingInfo.max_score || 0
						)}</span
					>
				</div>
			</div>
		{/if}
		{#if engine.trainingInfo.completed_games && engine.trainingInfo.num_games}
			<div class="w-full h-1.5 bg-surface-container-highest rounded-none overflow-hidden mt-1 border border-outline-variant/20">
				<div
					class="h-full bg-primary drop-shadow-[0_0_40px_var(--color-surface-tint)] transition-all duration-300"
					style="width: {(engine.trainingInfo.completed_games / engine.trainingInfo.num_games) *
						100}%"
				></div>
			</div>
		{/if}
	</div>
{/if}
