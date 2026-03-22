<script lang="ts">
	import { engine } from '../state.svelte.ts';
	
	let realMoves = $derived.by(() => {
		if (!engine.gameState?.mcts_mind) return [];
		
		const pieceNames = ['T', 'L', 'Z', 'I', 'O', 'S', 'J', 'vT', 'vL', 'vZ', 'vI', 'vO'];
		const totalVisits = engine.gameState.mcts_mind.reduce((sum: number, move: any) => sum + move.visits, 0) || 1;
		
		return engine.gameState.mcts_mind.map((move: any, i: number) => {
			const slot = Math.floor(move.action / 96);
			const pos = move.action % 96;
			const pieceName = pieceNames[engine.gameState.available[slot]] || `P-${engine.gameState.available[slot]}`;
			const confidence = Math.round((move.visits / totalVisits) * 100);
			
			const colors = [
				"bg-gradient-to-r from-primary to-primary-container drop-shadow-[0_0_40px_var(--color-surface-tint)]",
				"bg-primary/40",
				"bg-primary/20",
				"bg-primary/10",
				"bg-primary/5"
			];
			const color = colors[Math.min(i, colors.length - 1)];
			
			return {
				name: `Slot ${slot + 1} -> ${pieceName} -> Pos ${pos}`,
				confidence,
				color
			};
		});
	});
</script>

<div class="w-full flex flex-col lg:flex-row gap-6 mb-12 rounded-none">
<!-- Left Column: MCTS Mind-Reader & Death Trap -->
<div class="lg:w-1/3 flex flex-col gap-6">
<section class="bg-surface-bright/60 backdrop-blur-xl border-[0px] p-6 rounded-none">
<div class="flex items-center justify-between mb-8">
<h3 class="font-['Space_Grotesk'] text-sm font-bold tracking-widest text-primary uppercase">MCTS Mind-Reader</h3>
<span class="text-[10px] text-on-surface-variant font-mono">STEP_ID: 0x9AF</span>
</div>
<div class="space-y-6">
	{#each realMoves as move}
<!-- Move loop -->
<div class="group cursor-pointer">
<div class="flex justify-between items-end mb-2">
<span class="font-['Inter'] text-xs {move.confidence > 50 ? 'text-on-surface' : 'text-on-surface-variant'}">{move.name}</span>
<span class="font-['Space_Grotesk'] text-sm {move.confidence > 50 ? 'text-primary' : 'text-primary/50'} font-bold">{move.confidence}%</span>
</div>
<div class="h-1.5 w-full bg-surface-container-highest rounded-none overflow-hidden border border-outline-variant/20">
<div class="h-full {move.color}" style="width: {move.confidence}%"></div>
</div>
</div>
	{/each}
</div>
</section>
<!-- Death Trap Radar Module -->
<section class="bg-surface-bright/60 backdrop-blur-xl border-[0px] p-6 rounded-none border-error/50 border bg-error/5 flex items-center gap-6 overflow-hidden relative">
<div class="absolute inset-0 bg-gradient-to-r from-error/10 to-transparent pointer-events-none"></div>
<div class="relative w-16 h-16 shrink-0 rounded-none border border-error/50 flex items-center justify-center overflow-hidden">
<div class="absolute inset-0 bg-[conic-gradient(from_0deg,rgba(255,180,171,0.2),transparent_70%)] animate-[spin_4s_linear_infinite]"></div>
<div class="w-2 h-2 rounded-none bg-error drop-shadow-[0_0_8px_var(--color-error)]"></div>
<div class="absolute top-2 left-4 w-1 h-1 rounded-none bg-error/60"></div>
<div class="absolute bottom-4 right-3 w-1.5 h-1.5 rounded-none bg-error/80"></div>
</div>
<div class="flex flex-col gap-1 z-10">
<div class="flex items-center gap-2">
<span class="material-symbols-outlined text-error text-xl animate-pulse" style="font-variation-settings: 'FILL' 1;">warning</span>
<span class="font-['Space_Grotesk'] font-bold text-xs tracking-tighter text-error uppercase">SPATIAL THREAT DETECTED</span>
</div>
<p class="font-['Inter'] text-[10px] text-on-surface-variant leading-tight uppercase tracking-wide">Proximity alert: System suggests immediate node re-allocation to avoid structural deadlock.</p>
</div>
</section>
</div>

<!-- Right Column: Visualization and Controls -->
<div class="lg:w-2/3 flex flex-col gap-6">
<!-- Value vs Reality Sparkline Card -->
<section class="bg-surface-bright/60 backdrop-blur-xl border-[0px] p-8 rounded-none w-full">
<div class="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-12">
<div>
<h3 class="font-['Space_Grotesk'] text-sm font-bold tracking-widest text-primary uppercase mb-1">Value vs Reality</h3>
<p class="font-['Inter'] text-[10px] text-on-surface-variant uppercase tracking-widest">Temporal mapping of neural prediction vs outcome</p>
</div>
<div class="flex gap-12">
<div class="flex flex-col">
<span class="text-[10px] uppercase tracking-widest text-primary/60 font-medium">PREDICTED (V)</span>
<span class="font-['Space_Grotesk'] text-3xl font-bold text-primary">0.941</span>
</div>
<div class="flex flex-col">
<span class="text-[10px] uppercase tracking-widest text-secondary/60 font-medium">ACTUAL SCORE</span>
<span class="font-['Space_Grotesk'] text-3xl font-bold text-secondary">0.892</span>
</div>
</div>
</div>
<!-- Simulated Sparkline Chart -->
<div class="relative h-48 w-full border border-outline-variant/20 border-t-0 border-r-0">
<svg class="w-full h-full overflow-visible" viewBox="0 0 800 200" preserveAspectRatio="none">
<!-- Predicted Line (Cyan) -->
<path class="text-primary drop-shadow-[0_0_40px_var(--color-surface-tint)]" d="M0,140 Q100,60 200,100 T400,120 T600,40 T800,80" fill="none" stroke="currentColor" stroke-width="3"></path>
<path d="M0,140 Q100,60 200,100 T400,120 T600,40 T800,80 L800,200 L0,200 Z" fill="url(#gradient-primary)" fill-opacity="0.1"></path>
<!-- Actual Line (Emerald / Secondary) -->
<path class="text-secondary drop-shadow-[0_0_5px_var(--color-secondary)]" d="M0,150 Q120,110 240,140 T480,90 T720,130 T800,110" fill="none" stroke="currentColor" stroke-width="3" stroke-dasharray="8 4"></path>
<defs>
<linearGradient id="gradient-primary" x1="0" x2="0" y1="0" y2="1">
<stop offset="0%" stop-color="var(--color-primary)"></stop>
<stop offset="100%" stop-color="transparent"></stop>
</linearGradient>
</defs>
</svg>
<!-- Intersection Indicators -->
<div class="absolute left-1/4 top-[40%] w-3 h-3 rounded-none bg-primary border-2 border-background drop-shadow-[0_0_40px_var(--color-surface-tint)] animate-pulse"></div>
<div class="absolute left-3/4 top-[55%] w-3 h-3 rounded-none bg-secondary border-2 border-background drop-shadow-[0_0_8px_var(--color-secondary)] animate-pulse" style="animation-delay: 0.5s"></div>
</div>
</section>

<!-- Hyperparameter Tuning Console -->
<section class="bg-surface-bright/60 backdrop-blur-xl border-[0px] p-8 rounded-none w-full">
<h3 class="font-['Space_Grotesk'] text-sm font-bold tracking-widest text-primary uppercase mb-10">Hyperparameter Tuning Console</h3>
<div class="grid grid-cols-1 md:grid-cols-2 gap-12">
<!-- Slider 1 -->
<div class="space-y-6">
<div class="flex justify-between items-center">
<label class="font-['Inter'] text-[10px] font-bold text-on-surface uppercase tracking-widest">Temp Decay Steps</label>
<span class="font-['Space_Grotesk'] text-sm font-bold text-primary px-2 py-0.5 bg-primary/10 rounded-none border border-outline-variant/20">{engine.tempDecaySteps}</span>
</div>
<div class="relative group">
<div class="absolute -inset-1 bg-primary/20 blur opacity-0 group-hover:opacity-100 transition duration-500 pointer-events-none"></div>
<input class="relative w-full h-1.5 bg-surface-container-highest rounded-none border border-outline-variant/20 appearance-none cursor-pointer accent-primary" max="100" min="1" type="range" bind:value={engine.tempDecaySteps}/>
</div>
<div class="flex justify-between text-[10px] text-on-surface-variant font-mono uppercase">
<span>Greedy</span>
<span>Stochastic</span>
</div>
</div>
<!-- Slider 2 -->
<div class="space-y-6">
<div class="flex justify-between items-center">
<label class="font-['Inter'] text-[10px] font-bold text-on-surface uppercase tracking-widest">Gumbel Top-K Max</label>
<span class="font-['Space_Grotesk'] text-sm font-bold text-secondary px-2 py-0.5 bg-secondary/10 rounded-none border border-outline-variant/20">{engine.maxGumbelK}</span>
</div>
<div class="relative group">
<div class="absolute -inset-1 bg-secondary/20 blur opacity-0 group-hover:opacity-100 transition duration-500 pointer-events-none"></div>
<input class="relative w-full h-1.5 bg-surface-container-highest rounded-none border border-outline-variant/20 appearance-none cursor-pointer accent-secondary" max="16" min="4" type="range" bind:value={engine.maxGumbelK}/>
</div>
<div class="flex justify-between text-[10px] text-on-surface-variant font-mono uppercase">
<span>k=4</span>
<span>k=16</span>
</div>
</div>
</div>
</section>

</div>
</div>
