<script lang="ts">
	import { engine } from '$lib/state.svelte.ts';
</script>

{#if engine.isLeaderboardOpen}
	<div class="absolute inset-0 bg-zinc-950/95 z-30 p-8 overflow-y-auto flex flex-col">
		<div class="flex justify-between items-center mb-6">
			<h2 class="text-3xl font-black text-emerald-400 tracking-tight">Historic Replay Vault</h2>
			<button
				onclick={() => engine.toggleLeaderboard()}
				class="text-zinc-500 hover:text-rose-400 font-bold text-xl transition-colors">✕</button
			>
		</div>
		<div
			class="w-full text-left bg-zinc-900 rounded-xl overflow-hidden border border-zinc-800 shadow-xl"
		>
			<table class="w-full text-sm text-zinc-400">
				<thead
					class="bg-zinc-950 text-zinc-500 uppercase font-extrabold tracking-widest text-[10px] border-b border-zinc-800"
				>
					<tr>
						<th
							class="px-6 py-4 text-left cursor-pointer hover:text-white transition-colors"
							onclick={() => {
								engine.vaultSortKey = 'score';
								engine.vaultSortDesc = !engine.vaultSortDesc;
							}}
						>
							Efficiency {#if engine.vaultSortKey === 'score'}{engine.vaultSortDesc
									? '↓'
									: '↑'}{/if}
						</th>
						<th
							class="px-6 py-4 text-left cursor-pointer hover:text-white transition-colors"
							onclick={() => {
								engine.vaultSortKey = 'steps';
								engine.vaultSortDesc = !engine.vaultSortDesc;
							}}
						>
							Steps {#if engine.vaultSortKey === 'steps'}{engine.vaultSortDesc ? '↓' : '↑'}{/if}
						</th>
						<th
							class="px-6 py-4 text-left cursor-pointer hover:text-white transition-colors"
							onclick={() => {
								engine.vaultSortKey = 'difficulty';
								engine.vaultSortDesc = !engine.vaultSortDesc;
							}}
						>
							Complexity {#if engine.vaultSortKey === 'difficulty'}{engine.vaultSortDesc
									? '↓'
									: '↑'}{/if}
						</th>
						<th class="px-6 py-4 w-24 text-right">Action</th>
					</tr>
				</thead>
				<tbody>
					{#each engine.sortedTopGames as g}
						<tr class="border-b border-zinc-800/50 hover:bg-zinc-800/80 transition-colors">
							<td class="px-6 py-4 font-bold text-white text-lg"
								>{g.score}
								<span class="text-[10px] text-zinc-600 font-normal ml-2">ID: {g.id}</span></td
							>
							<td class="px-6 py-4 font-mono">{g.steps}</td>
							<td class="px-6 py-4 font-mono">{g.difficulty}</td>
							<td class="px-6 py-4 text-right">
								<button
									onclick={() => engine.replayGame(g.id)}
									class="text-emerald-400 hover:text-emerald-300 font-bold font-mono text-[10px] px-3 py-1.5 bg-emerald-500/10 rounded-md hover:bg-emerald-500/20 transition-all border border-emerald-500/20 shadow-[0_0_10px_rgba(16,185,129,0.15)] hover:shadow-[0_0_15px_rgba(16,185,129,0.3)]"
									>WATCH</button
								>
							</td>
						</tr>
					{/each}
					{#if engine.topGames.length === 0}
						<tr>
							<td colspan="3" class="px-6 py-12 text-center text-zinc-600 font-medium"
								>No historic simulations saved to SQLite.</td
							>
						</tr>
					{/if}
				</tbody>
			</table>
		</div>
	</div>
{/if}
