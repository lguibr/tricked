<script lang="ts">
	import { engine } from '$lib/state.svelte.js';
</script>

<div
	class="flex flex-wrap items-center justify-between gap-4 mb-8 bg-surface-bright/60 backdrop-blur-xl border-[0px] p-4 w-full max-w-3xl rounded-none border border-outline-variant/20"
>
	<div class="flex items-center gap-2">
		<span class="font-headline text-xs font-bold text-on-surface-variant uppercase tracking-widest mr-2">Complexity:</span>
		
		<button
			onclick={() => engine.resetGame(1)}
			disabled={engine.isTraining}
			class="px-5 py-2 text-xs font-headline rounded-none font-bold uppercase transition-all disabled:opacity-50 {engine.currentDifficulty === 1 ? 'bg-primary border border-primary text-on-primary drop-shadow-[0_0_40px_var(--color-surface-tint)]' : 'bg-surface-container text-primary border border-outline-variant/30 hover:bg-primary/20 hover:text-primary hover:border-primary/50'}"
			>1 - Easy</button
		>
		
		<button
			onclick={() => engine.resetGame(3)}
			disabled={engine.isTraining}
			class="px-5 py-2 text-xs font-headline rounded-none font-bold uppercase transition-all disabled:opacity-50 {engine.currentDifficulty === 3 ? 'bg-primary border border-primary text-on-primary drop-shadow-[0_0_40px_var(--color-surface-tint)]' : 'bg-surface-container text-primary border border-outline-variant/30 hover:bg-primary/20 hover:text-primary hover:border-primary/50'}"
			>3 - Normal</button
		>
		
		<button
			onclick={() => engine.resetGame(6)}
			disabled={engine.isTraining || engine.isReplaying}
			class="px-5 py-2 text-xs font-headline rounded-none font-bold uppercase transition-all disabled:opacity-50 {engine.currentDifficulty === 6 ? 'bg-secondary border border-secondary text-surface drop-shadow-[0_0_10px_var(--color-secondary)]' : 'bg-surface-container text-secondary border border-outline-variant/30 hover:bg-secondary/20 hover:text-secondary hover:border-secondary/50'}"
			>6 - Master</button
		>
	</div>
	
	<div class="flex items-center gap-3">
		<button
			onclick={() => engine.toggleLeaderboard()}
			disabled={engine.isTraining || engine.isReplaying}
			class="px-6 py-2 bg-surface-container text-on-surface text-xs rounded-none font-headline font-bold uppercase tracking-widest border border-outline-variant hover:text-primary hover:border-primary/50 transition-colors disabled:opacity-50 flex items-center gap-2"
		>
			<span class="material-symbols-outlined text-[16px]">history</span>
			Vault
		</button>
		
		<button
			onclick={() => engine.toggleTraining()}
			disabled={engine.isReplaying && engine.spectatorLastReplayedId === -1}
			class="flex items-center gap-2 px-6 py-2 text-xs font-headline rounded-none font-bold uppercase tracking-widest transition-all disabled:opacity-50 {engine.isTraining ? 'bg-secondary/20 text-secondary border border-secondary/50 animate-pulse' : 'bg-secondary border border-secondary text-surface hover:scale-[0.98] drop-shadow-[0_0_10px_var(--color-secondary)]' }"
		>
			<span class="material-symbols-outlined text-sm" style="font-variation-settings: 'FILL' 1;">{engine.isTraining ? 'pause' : 'play_arrow'}</span>
			{engine.isTraining ? 'PAUSE AI' : 'START CORE'}
		</button>
	</div>
</div>
