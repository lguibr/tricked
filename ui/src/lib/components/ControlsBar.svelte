<script lang="ts">
	import { engine } from '$lib/state.svelte.ts';
</script>

<div
	class="flex flex-col sm:flex-row gap-4 mb-8 bg-zinc-950/80 backdrop-blur-xl border border-zinc-800 p-3 items-center shadow-[0_0_30px_rgba(0,0,0,0.8)] w-full max-w-3xl justify-between rounded-none"
>
	<div class="flex items-center gap-2">
		<span class="text-xs font-bold uppercase tracking-widest text-zinc-500 ml-2 font-mono"
			>Complexity:</span
		>
		<button
			onclick={() => engine.resetGame(1)}
			disabled={engine.isTraining}
			class="px-4 py-1.5 text-sm font-mono font-medium rounded-none bg-zinc-800 hover:bg-zinc-700 text-zinc-300 transition-colors disabled:opacity-50"
			>1 (Easy)</button
		>
		<button
			onclick={() => engine.resetGame(3)}
			disabled={engine.isTraining}
			class="px-4 py-1.5 text-sm font-mono font-medium rounded-none bg-zinc-800 hover:bg-zinc-700 text-zinc-300 transition-colors disabled:opacity-50"
			>3 (Normal)</button
		>
		<button
			onclick={() => engine.resetGame(6)}
			disabled={engine.isTraining || engine.isReplaying}
			class="px-4 py-1.5 text-sm font-mono font-bold rounded-none bg-emerald-600/20 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-600/30 transition-all disabled:opacity-50 drop-shadow-[0_0_5px_rgba(16,185,129,0.3)]"
			>6 (Master)</button
		>
	</div>
	<div class="flex items-center gap-3">
		<a
			href="http://localhost:6006"
			target="_blank"
			class="px-5 py-2 rounded-none font-bold text-sm tracking-wide transition-all shadow-lg bg-zinc-800 text-orange-400 border border-orange-500/10 hover:bg-zinc-700 hover:border-orange-500/30 disabled:opacity-50 font-mono"
		>
			📊 TensorBoard
		</a>
		<button
			onclick={() => engine.toggleLeaderboard()}
			disabled={engine.isTraining || engine.isReplaying}
			class="px-5 py-2 rounded-none font-bold text-sm tracking-wide transition-all shadow-lg bg-zinc-800 text-emerald-400 border border-emerald-500/10 hover:bg-zinc-700 hover:border-emerald-500/30 disabled:opacity-50 font-mono"
		>
			🏆 Vault
		</button>
		<button
			onclick={() => engine.toggleTraining()}
			disabled={engine.isReplaying && engine.spectatorLastReplayedId === -1}
			class="px-6 py-2 rounded-none font-bold transition-all shadow-[0_0_15px_rgba(99,102,241,0.2)] hover:shadow-[0_0_20px_rgba(99,102,241,0.4)] disabled:opacity-50 font-mono {engine.isTraining
				? 'bg-amber-500/20 text-amber-500 border border-amber-500/30 animate-pulse'
				: 'bg-indigo-600/20 text-indigo-400 border border-indigo-500/30 hover:bg-indigo-600/30'}"
		>
			{engine.isTraining ? '⏸ Pause Training' : '▶ Start AI Model'}
		</button>
	</div>
</div>
