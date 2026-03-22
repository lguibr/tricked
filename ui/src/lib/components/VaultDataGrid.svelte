<script lang="ts">
	// @ts-expect-error Vite natively handles this but TS complains about extension
	import { engine } from '$lib/state.svelte.ts';
</script>

{#if engine.isLeaderboardOpen}
	<div class="fixed inset-0 bg-surface-dim/95 backdrop-blur-2xl z-50 p-6 flex flex-col items-center overflow-y-auto">
        <div class="w-full max-w-5xl h-full flex flex-col pt-10">
            <!-- Header section matches Stitch REPLAY_VAULT_DATABASE -->
            <div class="bg-surface-bright/60 backdrop-blur-xl border-[0px] flex-grow flex flex-col rounded-none border border-outline-variant/20">
                <div class="p-6 border-b border-outline-variant/30 flex flex-wrap items-center justify-between gap-4 bg-surface-container">
                    <div class="flex flex-col">
                        <h3 class="font-headline text-2xl font-black uppercase tracking-tighter text-primary drop-shadow-[0_0_40px_var(--color-surface-tint)] flex items-center gap-3">
                            <span class="material-symbols-outlined">database</span>
                            REPLAY_VAULT_DATABASE
                        </h3>
                        <p class="text-on-surface-variant text-xs font-bold tracking-widest uppercase mt-1 ml-9">Historic Simulations Extracted From Memory</p>
                    </div>

                    <!-- Filter Tabs -->
                    <div class="flex items-center gap-2">
                        <button onclick={() => {engine.fetchLeaderboard(null); engine.isLeaderboardOpen = true;}} class="px-4 py-2 font-headline font-bold text-xs rounded-none uppercase tracking-widest transition-colors {engine.vaultFilter === null ? 'bg-primary/20 text-primary border border-primary drop-shadow-[0_0_40px_var(--color-surface-tint)]' : 'bg-surface-container-highest border border-outline-variant/50 text-on-surface-variant hover:text-primary hover:border-primary/30'}">ALL GRADES</button>
                        <button onclick={() => engine.fetchLeaderboard(1)} class="px-4 py-2 font-headline font-bold text-xs rounded-none uppercase tracking-widest transition-colors {engine.vaultFilter === 1 ? 'bg-primary/20 text-primary border border-primary drop-shadow-[0_0_40px_var(--color-surface-tint)]' : 'bg-surface-container-highest border border-outline-variant/50 text-on-surface-variant hover:text-primary hover:border-primary/30'}">1 - EASY</button>
                        <button onclick={() => engine.fetchLeaderboard(3)} class="px-4 py-2 font-headline font-bold text-xs rounded-none uppercase tracking-widest transition-colors {engine.vaultFilter === 3 ? 'bg-primary/20 text-primary border border-primary drop-shadow-[0_0_40px_var(--color-surface-tint)]' : 'bg-surface-container-highest border border-outline-variant/50 text-on-surface-variant hover:text-primary hover:border-primary/30'}">3 - NORMAL</button>
                        <button onclick={() => engine.fetchLeaderboard(6)} class="px-4 py-2 font-headline font-bold text-xs rounded-none uppercase tracking-widest transition-colors {engine.vaultFilter === 6 ? 'bg-primary/20 text-primary border border-primary drop-shadow-[0_0_40px_var(--color-surface-tint)]' : 'bg-surface-container-highest border border-outline-variant/50 text-on-surface-variant hover:text-primary hover:border-primary/30'}">6 - MASTER</button>
                    </div>
                </div>
                
                <div class="flex-grow overflow-y-auto p-0 bg-surface-container-low">
                    <div class="text-[10px] font-headline text-on-surface-variant uppercase tracking-[0.2em] px-10 py-4 flex items-center justify-between border-b border-outline-variant/30">
                        <div class="flex items-center gap-12 sm:gap-24 w-1/2">
                            <span>RANK</span>
                            <span class="cursor-pointer hover:text-on-surface transition-colors" onclick={() => {engine.vaultSortKey = 'score'; engine.vaultSortDesc = !engine.vaultSortDesc;}}>SCORE_METRIC {#if engine.vaultSortKey === 'score'}{engine.vaultSortDesc ? '↓' : '↑'}{/if}</span>
                        </div>
                        <div class="flex items-center justify-between w-1/2">
                            <span class="cursor-pointer hover:text-on-surface transition-colors text-right flex-grow" onclick={() => {engine.vaultSortKey = 'steps'; engine.vaultSortDesc = !engine.vaultSortDesc;}}>TOTAL_STEPS {#if engine.vaultSortKey === 'steps'}{engine.vaultSortDesc ? '↓' : '↑'}{/if}</span>
                            <button onclick={() => engine.toggleLeaderboard()} class="ml-10 px-4 py-1.5 border border-error text-error font-bold hover:bg-error/20 hover:drop-shadow-[0_0_40px_var(--color-surface-tint)] transition-colors cursor-pointer text-[10px] rounded-none">CLOSE VAULT</button>
                        </div>
                    </div>

                    <!-- Games List -->
                    <div class="space-y-0 relative z-10 w-full">
                        {#each engine.sortedTopGames as g, index}
                            <!-- svelte-ignore a11y_click_events_have_key_events -->
                            <!-- svelte-ignore a11y_no_static_element_interactions -->
                            <div onclick={() => engine.replayGame(g.id)} class="group flex items-center justify-between px-10 py-5 bg-transparent border-b border-outline-variant/20 border-l-2 border-l-transparent hover:border-l-primary hover:bg-surface-container-high transition-all cursor-pointer">
                                <div class="flex items-center gap-12 sm:gap-24 w-1/2">
                                    <span class="font-headline text-2xl font-black text-outline group-hover:text-primary min-w-[30px] transition-colors">
                                        {(index + 1).toString().padStart(2, '0')}
                                    </span>
                                    <div>
                                        <div class="font-headline font-bold text-sm tracking-tighter uppercase group-hover:text-primary transition-colors text-on-surface">Game_Run_{g.id}</div>
                                        <div class="font-body text-[10px] text-on-surface-variant font-bold mt-1">Complexity: {g.difficulty} {g.difficulty === 1 ? 'Easy' : (g.difficulty === 3 ? 'Normal' : 'Master')}</div>
                                    </div>
                                    <div class="font-headline text-xl font-bold text-secondary">{g.score.toLocaleString()}</div>
                                </div>
                                <div class="text-right w-1/2 pr-32">
                                    <div class="font-headline font-bold text-sm text-on-surface uppercase">{g.steps} STEPS</div>
                                    <div class="font-body text-[10px] uppercase font-bold text-on-surface-variant mt-1 group-hover:text-primary transition-colors">PLAY_REPLAY →</div>
                                </div>
                            </div>
                        {/each}

                        {#if engine.topGames.length === 0}
                            <div class="p-20 text-center text-outline-variant font-bold uppercase tracking-widest text-sm flex flex-col items-center justify-center h-full">
                                <span class="material-symbols-outlined text-4xl mb-4 opacity-50">data_alert</span>
                                No historic simulations found in training database.
                            </div>
                        {/if}
                    </div>
                </div>
            </div>
        </div>
	</div>
{/if}
