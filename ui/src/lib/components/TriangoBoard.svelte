<script lang="ts">
	import { engine } from '$lib/state.svelte';
	import { getRowCol, isUp, getPoints, getBoardBit, getMaskBit } from '$lib/math';
</script>

<div
	class="relative aspect-square w-full max-w-3xl bg-surface-bright/60 backdrop-blur-xl border-[0px] p-0 overflow-hidden mb-8 group rounded-none border border-outline-variant/20"
>
	<div
		class="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-primary/10 via-transparent to-transparent pointer-events-none"
	></div>

	<svg
		class="w-[90%] h-[90%] filter drop-shadow-[0_0_25px_var(--color-primary-dim)] transition-all duration-500 mx-auto mt-[5%]"
		viewBox="-300 -300 600 600"
	>
		<g>
			{#each Array(96) as _, i}
				{@const [r, c] = getRowCol(i)}
				{@const up = isUp(r, c)}
				{@const points = getPoints(r, c, up)}
				{@const isPlaced = engine.gameState ? getBoardBit(engine.gameState.board, i) : false}
				{@const isHighlight = engine.activeMaskStr ? getMaskBit(engine.activeMaskStr, i) : false}
				{@const isDeathTrap = engine.gameState?.hole_logits
					? engine.gameState.hole_logits[i] > 0.5
					: false}

				<polygon
					{points}
					onclick={() => engine.handleClick(i)}
					role="button"
					tabindex="0"
					onkeydown={(e) => {
						if (e.key === 'Enter' || e.key === ' ') engine.handleClick(i);
					}}
					onmouseenter={() => (engine.hoveredIdx = i)}
					onmouseleave={() => {
						if (engine.hoveredIdx === i) engine.hoveredIdx = -1;
					}}
					class="cursor-pointer outline-none transition-all duration-200 transform-origin-center
					  {isHighlight
						? 'fill-primary opacity-90 stroke-primary-container stroke-[1.5px] z-10 scale-105 drop-shadow-[0_0_8px_var(--color-primary)]'
						: isDeathTrap
							? 'fill-error-container/80 stroke-error stroke-[2px] animate-pulse z-10 drop-shadow-[0_0_15px_var(--color-error)]'
							: isPlaced
								? 'fill-surface-container-highest/80 stroke-secondary stroke-[1px] drop-shadow-[0_0_5px_var(--color-secondary)]'
								: 'fill-black/40 hover:fill-surface-container-highest stroke-outline-variant/80 stroke-[1px]'}"
				/>
			{/each}
		</g>
	</svg>

	{#if engine.isReplaying && engine.replayStats}
		<div
			class="absolute bottom-0 left-0 w-full bg-surface-dim/90 backdrop-blur-xl border border-outline-variant/20 border-x-0 border-b-0 p-6 z-20"
		>
			<div class="flex items-center justify-between mb-6">
				<div class="flex items-center gap-4">
					<div
						class="px-3 py-1 bg-primary/10 border border-primary/20 text-primary font-headline text-[10px] tracking-widest uppercase font-bold rounded-none"
					>
						Simulation Mode
					</div>
					<span
						class="font-headline text-sm font-bold tracking-tighter text-on-surface uppercase flex items-center gap-3"
					>
						<span class="material-symbols-outlined text-outline text-lg">history</span>
						Step Counter: {engine.replayStats.currentStep} /
						<span class="text-outline">{engine.replayStats.maxStep}</span>
					</span>
				</div>
				<div class="flex items-center gap-4">
					<span class="font-headline text-sm font-bold tracking-tighter text-on-surface uppercase">
						Efficiency: <span class="text-secondary">{engine.gameState?.score}</span>
					</span>
					<button
						onclick={() => engine.stopReplay()}
						onkeydown={(e) => {
							if (e.key === 'Enter' || e.key === ' ') engine.stopReplay();
						}}
						class="px-5 py-2 rounded-none bg-error/10 text-error font-headline font-bold text-[10px] uppercase tracking-widest hover:bg-error hover:text-on-error border border-error/30 hover:border-error transition-colors"
						>EXIT REPLAY</button
					>
				</div>
			</div>
			<div class="space-y-2">
				<div
					class="flex justify-between text-[10px] font-headline text-on-surface-variant font-bold uppercase tracking-widest"
				>
					<span>Playback Speed</span>
					<span class="text-primary">{Math.round(1000 / engine.replaySpeedMs)} ops/sec</span>
				</div>
				<div
					class="relative h-1 bg-surface-container rounded-none overflow-visible border border-outline-variant/20"
				>
					<input
						type="range"
						min="50"
						max="1500"
						step="25"
						bind:value={engine.replaySpeedMs}
						class="absolute top-[-5px] left-0 w-full h-3 opacity-0 cursor-pointer z-10"
						style="direction: rtl;"
					/>

					<div
						class="absolute top-0 left-0 h-full bg-gradient-to-r from-primary to-secondary drop-shadow-[0_0_10px_var(--color-secondary)] flex items-center justify-end"
						style="width: {100 - ((engine.replaySpeedMs - 50) / 1450) * 100}%"
					>
						<div
							class="w-3 h-3 bg-white rounded-none shadow-[0_0_10px_rgba(255,255,255,1)] translate-x-1.5"
						></div>
					</div>
				</div>
			</div>
		</div>
	{/if}

	{#if engine.gameState?.terminal && !engine.isReplaying && !engine.isLeaderboardOpen}
		<div
			class="absolute inset-0 bg-surface/90 flex flex-col items-center justify-center p-12 text-center pointer-events-auto z-20"
		>
			<div class="mb-4 flex items-center gap-2 text-secondary">
				<span
					class="material-symbols-outlined text-4xl animate-pulse"
					style="font-variation-settings: 'FILL' 1;">warning</span
				>
				<span class="font-headline font-bold tracking-[0.4em] uppercase">Status: Terminal</span>
			</div>

			<h2
				class="font-headline text-5xl sm:text-7xl font-black text-on-surface uppercase mb-2 tracking-tighter drop-shadow-[0_0_10px_rgba(0,0,0,1)]"
			>
				SYSTEM HALTED
			</h2>
			<p class="font-headline text-secondary text-lg sm:text-xl uppercase tracking-widest mb-8">
				GAME OVER
			</p>
			<div class="grid grid-cols-2 gap-4 w-full max-w-md mb-8">
				<div
					class="p-4 rounded-none bg-surface-container-highest border border-outline-variant/30 border-l-4 border-l-secondary text-left drop-shadow-[0_0_40px_var(--color-surface-tint)]"
				>
					<div class="text-[10px] text-on-surface-variant font-bold uppercase tracking-widest mb-1">
						Final Efficiency
					</div>
					<div class="font-headline font-bold text-2xl text-secondary">
						{engine.gameState.score}
					</div>
				</div>
				<div
					class="p-4 rounded-none bg-surface-container-highest border border-outline-variant/30 border-l-4 border-l-primary text-left drop-shadow-[0_0_40px_var(--color-surface-tint)]"
				>
					<div class="text-[10px] text-on-surface-variant font-bold uppercase tracking-widest mb-1">
						Entities Cleared
					</div>
					<div class="font-headline font-bold text-2xl text-primary">
						{96 - (engine.gameState.pieces_left || 0)}/96
					</div>
				</div>
			</div>

			{#if !engine.isTraining}
				<button
					onclick={() => engine.resetGame(engine.currentDifficulty)}
					onkeydown={(e) => {
						if (e.key === 'Enter' || e.key === ' ') engine.resetGame(engine.currentDifficulty);
					}}
					class="px-10 py-4 rounded-none bg-secondary/10 border border-secondary text-secondary font-bold font-headline uppercase tracking-widest text-sm hover:bg-secondary hover:text-surface transition-all hover:scale-105 drop-shadow-[0_0_15px_var(--color-secondary)]"
					>Relaunch Engine</button
				>
			{:else}
				<div
					class="px-8 py-3 rounded-none border border-secondary/30 text-secondary font-bold uppercase tracking-widest text-xs animate-pulse bg-secondary/10"
				>
					Self-Optimizing Loop...
				</div>
			{/if}
		</div>
	{/if}
</div>
