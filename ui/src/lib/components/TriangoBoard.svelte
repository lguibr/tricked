<script lang="ts">
	import { engine } from '$lib/state.svelte.ts';
	import { getRowCol, isUp, getPoints, getBoardBit, getMaskBit } from '$lib/math';
</script>

<div
	class="relative w-full max-w-3xl aspect-square bg-zinc-950 rounded-none shadow-[0_0_40px_rgba(0,0,0,0.8)] flex items-center justify-center border border-zinc-800 mb-8 overflow-hidden group"
>
	<svg
		class="w-[90%] h-[90%] filter drop-shadow-[0_0_25px_rgba(16,185,129,0.15)] transition-all duration-500"
		viewBox="-300 -300 600 600"
	>
		<g>
			{#each Array(96) as _, i}
				{@const [r, c] = getRowCol(i)}
				{@const up = isUp(r, c)}
				{@const points = getPoints(r, c, up)}
				{@const isPlaced = engine.gameState ? getBoardBit(engine.gameState.board, i) : false}
				{@const isHighlight = engine.activeMaskStr ? getMaskBit(engine.activeMaskStr, i) : false}

				<!-- svelte-ignore a11y_click_events_have_key_events -->
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<polygon
					{points}
					onclick={() => engine.handleClick(i)}
					onmouseenter={() => (engine.hoveredIdx = i)}
					onmouseleave={() => {
						if (engine.hoveredIdx === i) engine.hoveredIdx = -1;
					}}
					class="cursor-pointer outline-none transition-all duration-200 transform-origin-center
					  {isHighlight
						? 'fill-emerald-400 opacity-90 stroke-emerald-200 stroke-[1.5px] z-10 scale-105 drop-shadow-[0_0_10px_rgba(16,185,129,0.8)]'
						: isPlaced
							? 'fill-zinc-900 stroke-emerald-900/40 stroke-[1px]'
							: 'fill-[#0a0a0a] hover:fill-zinc-900 stroke-zinc-800/80 stroke-[1px]'}"
				/>
			{/each}
		</g>
	</svg>

	<!-- Replay active badge -->
	{#if engine.isReplaying && engine.replayStats}
		<div
			class="absolute top-6 left-0 right-0 flex justify-center items-start z-20 pointer-events-none flex-col items-center gap-2"
		>
			<div
				class="px-6 py-2 bg-indigo-500/20 backdrop-blur-xl border border-indigo-500/40 text-indigo-200 uppercase tracking-widest font-bold text-[11px] rounded-none shadow-2xl pointer-events-auto flex items-center gap-4 font-mono"
			>
				<span>Historic Simulation</span>
				<div
					class="flex items-center gap-4 border-l border-indigo-500/30 pl-4 h-full bg-indigo-500/10 rounded-none py-1 pr-6"
				>
					<span class="text-white flex items-center gap-2"
						>Difficulty <span
							class="px-2 py-0.5 rounded-md bg-amber-500/20 text-amber-400 border-amber-500/30 border"
							>{engine.gameState?.difficulty}</span
						></span
					>
					<span class="text-white border-l border-indigo-500/30 pl-4 h-full flex items-center"
						>Step <span class="text-indigo-400 ml-1">{engine.replayStats.currentStep}</span> /
						<span class="text-zinc-400 ml-1">{engine.replayStats.maxStep}</span></span
					>
					<span class="text-white border-l border-indigo-500/30 pl-4 h-full flex items-center"
						>Efficiency <span class="text-emerald-400 ml-1">{engine.gameState?.score}</span> /
						<span class="text-zinc-400 ml-1">{engine.replayStats.maxScore}</span></span
					>
				</div>
				<button
					onclick={() => engine.stopReplay()}
					class="ml-2 bg-indigo-500 text-zinc-950 px-4 py-1.5 rounded-full font-black hover:scale-105 transition-transform shadow-[0_0_15px_rgba(99,102,241,0.4)]"
					>EXIT</button
				>
			</div>

			{#if engine.isReplaying}
				<!-- Slider overlay -->
				<div
					class="px-5 py-3 mt-1 bg-zinc-900/90 backdrop-blur-xl border border-zinc-700/50 rounded-2xl shadow-2xl pointer-events-auto flex flex-col items-center gap-2 min-w-[340px] transition-all"
				>
					<span
						class="text-[10px] text-zinc-400 font-bold uppercase tracking-widest flex items-center gap-2"
						>Playback Speed: <span class="text-white"
							>{Math.round(1000 / engine.replaySpeedMs)} ops/sec</span
						>
						<span class="text-amber-500 bg-amber-500/10 px-1.5 rounded border border-amber-500/20"
							>Game #{engine.replayStats.gameId}</span
						></span
					>
					<input
						type="range"
						min="50"
						max="1500"
						step="25"
						bind:value={engine.replaySpeedMs}
						class="w-full h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
						style="direction: rtl;"
					/>
					<div
						class="flex justify-between w-full text-[9px] text-zinc-400 font-mono font-bold px-1 mt-1"
					>
						<span>Hyperspeed</span>
						<span>Cinematic</span>
					</div>
				</div>
			{/if}
		</div>
	{/if}

	<!-- Game Over Overlay -->
	{#if engine.gameState?.terminal && !engine.isReplaying && !engine.isLeaderboardOpen}
		<div
			class="absolute inset-0 bg-zinc-950/20 backdrop-blur-[2px] flex flex-col items-center justify-center z-20 pointer-events-none"
		>
			<div
				class="bg-zinc-950/90 py-6 px-10 rounded-none border border-rose-500/30 shadow-[0_0_50px_rgba(244,63,94,0.3)] flex flex-col items-center pointer-events-auto"
			>
				<h2 class="text-4xl font-black text-rose-500 mb-2 drop-shadow-2xl">SYSTEM HALTED</h2>
				<p class="text-lg text-zinc-300 mb-6">
					Final Efficiency: <span class="font-bold text-white tracking-widest"
						>{engine.gameState.score}</span
					>
				</p>
				{#if !engine.isTraining}
					<button
						onclick={() => engine.resetGame()}
						class="px-8 py-3 bg-emerald-500 text-zinc-950 rounded-full font-black shadow-[0_0_20px_rgba(16,185,129,0.3)] hover:scale-105 transition-transform"
						>Re-Initialize</button
					>
				{:else}
					<div
						class="px-8 py-3 bg-indigo-500/20 text-indigo-300 rounded-full font-black animate-pulse"
					>
						Training Loop...
					</div>
				{/if}
			</div>
		</div>
	{/if}
</div>
