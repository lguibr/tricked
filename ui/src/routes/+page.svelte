<script lang="ts">
	import { onMount, onDestroy } from 'svelte';

	// Game Constants
	const ROW_LENGTHS = [9, 11, 13, 15, 15, 13, 11, 9];
	const TOTAL_TRIANGLES = 96;
	const TRI_SIDE = 40;
	const TRI_HEIGHT = 34.64; // 40 * sin(60)

	interface GameStateData {
		board: string;
		piece_masks: Record<string, string[]>;
		available: number[];
		score: number;
		pieces_left: number;
		terminal: boolean;
		difficulty: number;
	}

	// Reactive State
	let gameState = $state<GameStateData | null>(null);
	let selectedSlot = $state(-1);
	let hoveredIdx = $state(-1);
	let isTraining = $state(false);
	let spectatorInterval: ReturnType<typeof setInterval> | null = null;
	let loading = $state(true);

	// Vault & Spectator State
	let isLeaderboardOpen = $state(false);
	let topGames = $state<any[]>([]);
	let isReplaying = $state(false);
	let replayInterval: ReturnType<typeof setInterval> | null = null;
	let spectatorLastReplayedId = $state(-1);
	let replayStats = $state<any>(null);

	// Training Status Telemetry
	let trainingInfo = $state<any>(null);
	let trainingStatusInterval: ReturnType<typeof setInterval> | null = null;

	// Math Utilities
	function getRowCol(idx: number): [number, number] {
		let rem = idx;
		for (let r = 0; r < 8; r++) {
			if (rem < ROW_LENGTHS[r]) return [r, rem];
			rem -= ROW_LENGTHS[r];
		}
		return [-1, -1];
	}

	function isUp(r: number, c: number): boolean {
		if (r < 4) return c % 2 === 0;
		return c % 2 === 1;
	}

	function getPoints(r: number, c: number, isUpTri: boolean): string {
		const rowOffset = (15 - ROW_LENGTHS[r]) * (TRI_SIDE / 4);
		const x = c * (TRI_SIDE / 2) + rowOffset - 140;
		const y = r * TRI_HEIGHT - 130;

		if (isUpTri) {
			return `${x},${y + TRI_HEIGHT} ${x + TRI_SIDE / 2},${y} ${x + TRI_SIDE},${y + TRI_HEIGHT}`;
		} else {
			return `${x},${y} ${x + TRI_SIDE / 2},${y + TRI_HEIGHT} ${x + TRI_SIDE},${y}`;
		}
	}

	function getBoardBit(boardStr: string, idx: number): boolean {
		const board = BigInt(boardStr);
		const mask = 1n << BigInt(idx);
		return (board & mask) !== 0n;
	}

	function getMaskBit(maskStr: string, idx: number): boolean {
		const mask = BigInt(maskStr);
		const b = 1n << BigInt(idx);
		return (mask & b) !== 0n;
	}

	function findValidPlacementIndex(p_id: number, anchorIdx: number): number {
		if (!gameState || p_id === -1) return -1;

		const masks = gameState.piece_masks[p_id];
		for (let idx = 0; idx < 96; idx++) {
			const mStr = masks[idx];
			if (mStr === '0') continue;

			if (getMaskBit(mStr, anchorIdx)) {
				const currentBoard = BigInt(gameState.board);
				const moveMask = BigInt(mStr);
				if ((currentBoard & moveMask) === 0n) {
					return idx;
				}
			}
		}
		return -1;
	}

	// Derived Reactive Data
	let activeMaskStr = $derived.by(() => {
		if (selectedSlot === -1 || !gameState || hoveredIdx === -1) return null;
		const p_id = gameState.available[selectedSlot];
		if (p_id === -1) return null;
		const validIdx = findValidPlacementIndex(p_id, hoveredIdx);
		if (validIdx !== -1) {
			return gameState.piece_masks[p_id][validIdx];
		}
		return null;
	});

	// Tray Piece Calculation
	function getPieceData(p_id: number) {
		if (p_id === -1 || !gameState) return null;
		let mStr = '0';
		for (let idx = 0; idx < 96; idx++) {
			if (gameState.piece_masks[p_id][idx] !== '0') {
				mStr = gameState.piece_masks[p_id][idx];
				break;
			}
		}
		if (mStr === '0') return null;

		let polys: string[] = [];
		let minX = 999,
			minY = 999,
			maxX = -999,
			maxY = -999;

		for (let i = 0; i < TOTAL_TRIANGLES; i++) {
			if (getMaskBit(mStr, i)) {
				const [r, c] = getRowCol(i);
				const up = isUp(r, c);
				const pts = getPoints(r, c, up);

				const rowOffset = (15 - ROW_LENGTHS[r]) * (TRI_SIDE / 4);
				const x = c * (TRI_SIDE / 2) + rowOffset - 140;
				const y = r * TRI_HEIGHT - 130;

				if (x < minX) minX = x;
				if (x > maxX) maxX = x;
				if (y < minY) minY = y;
				if (y > maxY) maxY = y;

				polys.push(pts);
			}
		}
		const cx = minX + (maxX - minX + TRI_SIDE) / 2;
		const cy = minY + (maxY - minY + TRI_HEIGHT) / 2;
		return { polys, viewBox: `${cx - 60} ${cy - 60} 120 120` };
	}

	// API Interactions
	const API_BASE = 'http://127.0.0.1:8080/api';

	async function fetchState() {
		try {
			const res = await fetch(`${API_BASE}/state`);
			if (res.ok) gameState = await res.json();
		} catch (e) {
			console.error('Could not reach API');
		}
		loading = false;
	}

	async function resetGame(diff = 6) {
		const res = await fetch(`${API_BASE}/reset`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ difficulty: diff })
		});
		if (res.ok) {
			gameState = await res.json();
			selectedSlot = -1;
		}
	}

	async function rotateSlot(s: number) {
		if (!gameState || gameState.available[s] === -1 || isTraining) return;
		const res = await fetch(`${API_BASE}/rotate`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ slot: s })
		});
		if (res.ok) gameState = await res.json();
	}

	async function handleClick(anchorIdx: number) {
		if (selectedSlot === -1 || !gameState || isTraining) return;
		const p_id = gameState.available[selectedSlot];
		if (p_id === -1) return;

		const validIdx = findValidPlacementIndex(p_id, anchorIdx);
		if (validIdx !== -1) {
			const res = await fetch(`${API_BASE}/move`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ slot: selectedSlot, idx: validIdx })
			});
			if (res.ok) {
				gameState = await res.json();
				selectedSlot = -1;
			}
		}
	}

	function startSpectatorPolling() {
		if (spectatorInterval) return;
		spectatorInterval = setInterval(async () => {
			try {
				const res = await fetch(`${API_BASE}/games/top`);
				if (res.ok) {
					const games = await res.json();
					topGames = games; // Bind for the slider
					if (games.length > 0) {
						const topGame = games[0];
						// If a NEW higher score is found, instantly broadcast and restart
						if (topGame.id !== spectatorLastReplayedId) {
							spectatorLastReplayedId = topGame.id;
							replayGame(topGame.id, true);
						}
						// Alternatively, if it's the SAME game but it finished visually, loop it.
						else if (!isReplaying) {
							replayGame(topGame.id, true);
						}
					}
				}
			} catch (e) {}
		}, 3000);
	}

	async function checkTrainingStatus() {
		try {
			const res = await fetch(`${API_BASE}/training/status`);
			if (res.ok) {
				const data = await res.json();
				trainingInfo = data;
				if (data.running && !isTraining) {
					isTraining = true;
					startSpectatorPolling();
				} else if (!data.running && isTraining) {
					isTraining = false;
					if (spectatorInterval) {
						clearInterval(spectatorInterval);
						spectatorInterval = null;
					}
					stopReplay();
				}
			}
		} catch (e) {}
	}

	async function toggleTraining() {
		if (isTraining) {
			isTraining = false;
			if (spectatorInterval) {
				clearInterval(spectatorInterval);
				spectatorInterval = null;
			}
			stopReplay();
			spectatorLastReplayedId = -1;
			await fetch(`${API_BASE}/training/stop`, { method: 'POST' });
			fetchState();
		} else {
			isTraining = true;
			await fetch(`${API_BASE}/training/start`, { method: 'POST' });
			startSpectatorPolling();
		}
	}

	// Vault Handlers
	async function toggleLeaderboard() {
		isLeaderboardOpen = !isLeaderboardOpen;
		if (isLeaderboardOpen) {
			const res = await fetch(`${API_BASE}/games/top`);
			if (res.ok) topGames = await res.json();
		} else {
			stopReplay();
		}
	}

	function stopReplay() {
		isReplaying = false;
		replayStats = null;
		if (replayInterval) {
			clearInterval(replayInterval);
			replayInterval = null;
		}
		fetchState();
	}

	async function replayGame(id: number, isSpectatorLoop = false) {
		stopReplay();
		const res = await fetch(`${API_BASE}/games/${id}`);
		if (res.ok) {
			const data = await res.json();
			const moves = data.moves;
			isLeaderboardOpen = false;
			isReplaying = true;

			let step = 0;
			const maxStep = moves.length;
			if (gameState) {
				gameState.score = data.score;
				gameState.difficulty = data.difficulty;
				gameState.pieces_left = 96 - data.steps;
				gameState.terminal = true;
			}

			replayInterval = setInterval(() => {
				if (step < maxStep && gameState) {
					const moveData = moves[step];
					if (typeof moveData === 'string') {
						gameState.board = moveData;
					} else {
						gameState.board = moveData.board;
						gameState.score = moveData.score;
						gameState.available = moveData.available;
					}

					replayStats = {
						currentStep: step + 1,
						maxStep,
						gameId: id,
						maxScore: data.score
					};
					step++;
				} else {
					if (replayInterval) {
						clearInterval(replayInterval);
						replayInterval = null;
					}
					if (isSpectatorLoop && isTraining) {
						setTimeout(() => {
							if (isTraining && isReplaying) {
								replayGame(id, true);
							} else {
								stopReplay();
							}
						}, 2000);
					} else {
						stopReplay();
					}
				}
			}, 750); // Slower animation
		}
	}

	onMount(() => {
		fetchState();
		checkTrainingStatus();
		trainingStatusInterval = setInterval(checkTrainingStatus, 1500);
	});

	onDestroy(() => {
		if (spectatorInterval) clearInterval(spectatorInterval);
		if (trainingStatusInterval) clearInterval(trainingStatusInterval);
	});
</script>

<svelte:head>
	<title>Tricked: AI Engine</title>
</svelte:head>

<main class="min-h-screen flex flex-col items-center justify-center p-8 font-sans">
	{#if loading}
		<div class="text-emerald-500 animate-pulse text-2xl font-bold">
			Booting Svelte Neural Interface...
		</div>
	{:else}
		<!-- Header -->
		<div class="mb-6 text-center">
			<h1
				class="text-5xl font-black tracking-tight text-transparent bg-clip-text bg-gradient-to-br from-emerald-400 to-cyan-500 mb-2 drop-shadow-sm"
			>
				Tricked
			</h1>
			<p class="text-zinc-500 font-medium tracking-wide">120-Degree Mathematical Engine</p>
		</div>

		<!-- Controls -->
		<div
			class="flex flex-col sm:flex-row gap-4 mb-8 bg-zinc-900/80 backdrop-blur border border-zinc-800 p-3 rounded-2xl items-center shadow-2xl w-full max-w-3xl justify-between"
		>
			<div class="flex items-center gap-2">
				<span class="text-xs font-bold uppercase tracking-widest text-zinc-500 ml-2"
					>Complexity:</span
				>
				<button
					onclick={() => resetGame(1)}
					disabled={isTraining}
					class="px-4 py-1.5 text-sm font-medium rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 transition-colors disabled:opacity-50"
					>1 (Easy)</button
				>
				<button
					onclick={() => resetGame(3)}
					disabled={isTraining}
					class="px-4 py-1.5 text-sm font-medium rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 transition-colors disabled:opacity-50"
					>3 (Normal)</button
				>
				<button
					onclick={() => resetGame(6)}
					disabled={isTraining || isReplaying}
					class="px-4 py-1.5 text-sm font-bold rounded-lg bg-emerald-600/20 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-600/30 transition-all disabled:opacity-50"
					>6 (Master)</button
				>
			</div>
			<div class="flex items-center gap-3">
				<a
					href="http://localhost:6006"
					target="_blank"
					class="px-5 py-2 rounded-xl font-bold text-sm tracking-wide transition-all shadow-lg bg-zinc-800 text-orange-400 border border-orange-500/10 hover:bg-zinc-700 hover:border-orange-500/30 disabled:opacity-50"
				>
					📊 TensorBoard
				</a>
				<button
					onclick={toggleLeaderboard}
					disabled={isTraining || isReplaying}
					class="px-5 py-2 rounded-xl font-bold text-sm tracking-wide transition-all shadow-lg bg-zinc-800 text-emerald-400 border border-emerald-500/10 hover:bg-zinc-700 hover:border-emerald-500/30 disabled:opacity-50"
				>
					🏆 Vault
				</button>
				<button
					onclick={toggleTraining}
					disabled={isReplaying && spectatorLastReplayedId === -1}
					class="px-6 py-2 rounded-xl font-bold transition-all shadow-lg disabled:opacity-50 {isTraining
						? 'bg-amber-500/20 text-amber-500 border border-amber-500/30 animate-pulse'
						: 'bg-indigo-600/20 text-indigo-400 border border-indigo-500/30 hover:bg-indigo-600/30'}"
				>
					{isTraining ? '⏸ Pause Training' : '▶ Start AI Model'}
				</button>
			</div>
		</div>

		<!-- Board Area -->
		<div
			class="relative w-full max-w-3xl aspect-square bg-zinc-900 rounded-3xl shadow-2xl flex items-center justify-center border border-zinc-800 mb-8 overflow-hidden group"
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
						{@const isPlaced = gameState ? getBoardBit(gameState.board, i) : false}
						{@const isHighlight = activeMaskStr ? getMaskBit(activeMaskStr, i) : false}

						<!-- svelte-ignore a11y_click_events_have_key_events -->
						<!-- svelte-ignore a11y_no_static_element_interactions -->
						<polygon
							{points}
							onclick={() => handleClick(i)}
							onmouseenter={() => (hoveredIdx = i)}
							onmouseleave={() => {
								if (hoveredIdx === i) hoveredIdx = -1;
							}}
							class="cursor-pointer outline-none transition-all duration-200 transform-origin-center
                              {isHighlight
								? 'fill-emerald-400 opacity-90 stroke-emerald-200 stroke-[2px] z-10 scale-105'
								: isPlaced
									? 'fill-zinc-700 stroke-zinc-950 stroke-[2px]'
									: 'fill-zinc-800 hover:fill-zinc-700 stroke-zinc-700 stroke-[1px]'}"
						/>
					{/each}
				</g>
			</svg>

			<!-- Replay active badge -->
			{#if isReplaying && replayStats}
				<div
					class="absolute top-6 left-0 right-0 flex justify-center items-start z-20 pointer-events-none flex-col items-center gap-2"
				>
					<div
						class="px-6 py-2 bg-indigo-500/20 backdrop-blur-xl border border-indigo-500/40 text-indigo-200 uppercase tracking-widest font-bold text-[11px] rounded-full shadow-2xl pointer-events-auto flex items-center gap-4"
					>
						<span>Historic Simulation</span>
						<div
							class="flex items-center gap-4 border-l border-indigo-500/30 pl-4 h-full bg-indigo-500/10 rounded-r-full py-1 pr-6"
						>
							<span class="text-white flex items-center gap-2"
								>Difficulty <span
									class="px-2 py-0.5 rounded-md bg-amber-500/20 text-amber-400 border-amber-500/30 border"
									>{gameState?.difficulty}</span
								></span
							>
							<span class="text-white border-l border-indigo-500/30 pl-4 h-full flex items-center"
								>Step <span class="text-indigo-400 ml-1">{replayStats.currentStep}</span> /
								<span class="text-zinc-400 ml-1">{replayStats.maxStep}</span></span
							>
							<span class="text-white border-l border-indigo-500/30 pl-4 h-full flex items-center"
								>Efficiency <span class="text-emerald-400 ml-1">{gameState?.score}</span> /
								<span class="text-zinc-400 ml-1">{replayStats.maxScore}</span></span
							>
						</div>
						<button
							onclick={stopReplay}
							class="ml-2 bg-indigo-500 text-zinc-950 px-4 py-1.5 rounded-full font-black hover:scale-105 transition-transform shadow-[0_0_15px_rgba(99,102,241,0.4)]"
							>EXIT</button
						>
					</div>

					{#if isTraining && topGames.length > 0}
						<!-- Slider overlay -->
						<div
							class="px-5 py-3 mt-1 bg-zinc-900/90 backdrop-blur-xl border border-zinc-700/50 rounded-2xl shadow-2xl pointer-events-auto flex flex-col items-center gap-2 min-w-[340px] transition-all"
						>
							<span
								class="text-[10px] text-zinc-400 font-bold uppercase tracking-widest flex items-center gap-2"
								>Simulation Target: <span class="text-white">Game #{replayStats.gameId}</span>
								<span
									class="text-amber-500 bg-amber-500/10 px-1.5 rounded border border-amber-500/20"
									>Diff {gameState?.difficulty}</span
								></span
							>
							<input
								type="range"
								min="0"
								max={Math.min(9, topGames.length - 1)}
								value={topGames.findIndex((g) => g.id === replayStats.gameId) === -1
									? 0
									: topGames.findIndex((g) => g.id === replayStats.gameId)}
								onchange={(e) => {
									const idx = parseInt(e.currentTarget.value);
									const targetGame = topGames[idx];
									if (targetGame && targetGame.id !== replayStats.gameId) {
										spectatorLastReplayedId = targetGame.id;
										replayGame(targetGame.id, true);
									}
								}}
								class="w-full h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
							/>
							<div
								class="flex justify-between w-full text-[9px] text-zinc-400 font-mono font-bold px-1 mt-1"
							>
								<span>Top #1 (Diff {topGames[0]?.difficulty})</span>
								<span
									>Top #{Math.min(10, topGames.length)} (Diff {topGames[
										Math.min(9, topGames.length - 1)
									]?.difficulty})</span
								>
							</div>
						</div>
					{/if}
				</div>
			{/if}

			<!-- Vault Overlay -->
			{#if isLeaderboardOpen}
				<div class="absolute inset-0 bg-zinc-950/95 z-30 p-8 overflow-y-auto flex flex-col">
					<div class="flex justify-between items-center mb-6">
						<h2 class="text-3xl font-black text-emerald-400 tracking-tight">
							Historic Replay Vault
						</h2>
						<button
							onclick={toggleLeaderboard}
							class="text-zinc-500 hover:text-rose-400 font-bold text-xl transition-colors"
							>✕</button
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
									<th class="px-6 py-4 text-left">Efficiency</th>
									<th class="px-6 py-4 text-left">Complexity</th>
									<th class="px-6 py-4 w-24 text-right">Action</th>
								</tr>
							</thead>
							<tbody>
								{#each topGames as g}
									<tr class="border-b border-zinc-800/50 hover:bg-zinc-800/80 transition-colors">
										<td class="px-6 py-4 font-bold text-white text-lg"
											>{g.score}
											<span class="text-[10px] text-zinc-600 font-normal ml-2">{g.steps} STPS</span
											></td
										>
										<td class="px-6 py-4 font-mono">{g.difficulty}</td>
										<td class="px-6 py-4 text-right">
											<button
												onclick={() => replayGame(g.id)}
												class="text-emerald-400 hover:text-emerald-300 font-bold font-mono text-[10px] px-3 py-1.5 bg-emerald-500/10 rounded-md hover:bg-emerald-500/20 transition-all border border-emerald-500/20"
												>WATCH</button
											>
										</td>
									</tr>
								{/each}
								{#if topGames.length === 0}
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

			<!-- Game Over Overlay -->
			{#if gameState?.terminal && !isReplaying && !isLeaderboardOpen}
				<div
					class="absolute inset-0 bg-zinc-950/20 backdrop-blur-[2px] flex flex-col items-center justify-center z-20 pointer-events-none"
				>
					<div
						class="bg-zinc-950/90 py-6 px-10 rounded-3xl border border-rose-500/30 shadow-2xl flex flex-col items-center pointer-events-auto"
					>
						<h2 class="text-4xl font-black text-rose-500 mb-2 drop-shadow-2xl">SYSTEM HALTED</h2>
						<p class="text-lg text-zinc-300 mb-6">
							Final Efficiency: <span class="font-bold text-white tracking-widest"
								>{gameState.score}</span
							>
						</p>
						{#if !isTraining}
							<button
								onclick={() => resetGame()}
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

		<!-- Footer Stats -->
		<div class="w-full max-w-3xl flex justify-between items-end mb-6 px-4">
			<div class="text-3xl font-bold tracking-tight text-zinc-200">
				Efficiency: <span class="text-emerald-400">{gameState?.score || 0}</span>
			</div>
			<div class="text-zinc-500 font-medium">
				Entities Remaining: <span class="text-zinc-300">{gameState?.pieces_left || 0}</span>
			</div>
		</div>

		<!-- Piece Trays -->
		<div class="grid grid-cols-3 gap-6 w-full max-w-3xl">
			{#each [0, 1, 2] as s}
				{@const p_id = gameState ? gameState.available[s] : -1}
				{@const isSelected = selectedSlot === s}
				{@const pieceData = getPieceData(p_id)}

				<!-- svelte-ignore a11y_click_events_have_key_events -->
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div
					class="bg-zinc-900/50 rounded-2xl border {isSelected
						? 'border-emerald-500 bg-emerald-500/5 scale-105'
						: 'border-zinc-800'} p-4 shadow-inner min-h-[160px] flex items-center justify-center flex-col relative cursor-pointer transition-all duration-200 hover:border-zinc-600"
					onclick={() => {
						if (!isTraining && p_id !== -1) selectedSlot = isSelected ? -1 : s;
					}}
					oncontextmenu={(e) => {
						e.preventDefault();
						rotateSlot(s);
					}}
				>
					<span
						class="absolute top-3 left-3 text-[10px] font-bold uppercase tracking-widest text-zinc-600"
						>Buffer {s}</span
					>
					{#if pieceData}
						<!-- Piece Render -->
						<svg
							class="w-full h-full drop-shadow-[0_0_12px_rgba(16,185,129,0.3)]"
							viewBox={pieceData.viewBox}
						>
							{#each pieceData.polys as pts}
								<polygon points={pts} class="fill-emerald-500 stroke-emerald-300 stroke-[2px]" />
							{/each}
						</svg>
					{/if}
				</div>
			{/each}
		</div>

		<div class="text-center mt-6 text-zinc-500 text-xs font-semibold tracking-widest uppercase">
			Right-Click to mathematically rotate buffer entities 60°
		</div>
	{/if}

	{#if isTraining && trainingInfo}
		<div
			class="fixed top-6 right-6 z-50 bg-zinc-900/90 backdrop-blur-xl border border-zinc-800 shadow-2xl p-4 rounded-2xl flex flex-col gap-2 min-w-[280px]"
		>
			<div class="flex items-center justify-between mb-1">
				<span
					class="text-[10px] font-black tracking-widest uppercase text-amber-500 animate-pulse flex items-center gap-2"
					><span class="w-2 h-2 rounded-full bg-amber-500"></span> Training Active</span
				>
			</div>
			{#if trainingInfo.iteration}
				<div class="flex flex-col gap-1 items-start text-left w-full">
					<span class="text-xs font-mono font-bold text-zinc-300 w-full mb-2"
						>{trainingInfo.stage || 'Initializing...'}</span
					>
					<div
						class="flex justify-between w-full text-[10px] text-zinc-500 font-bold uppercase tracking-widest"
					>
						<span>Epoch {trainingInfo.iteration}/{trainingInfo.total_iterations}</span>
						<span class="text-emerald-400"
							>Med: {Math.round(trainingInfo.median_score || 0)} / Max: {Math.round(
								trainingInfo.max_score || 0
							)}</span
						>
					</div>
				</div>
			{/if}
			{#if trainingInfo.completed_games && trainingInfo.num_games}
				<div class="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden mt-1">
					<div
						class="h-full bg-amber-500 transition-all duration-300"
						style="width: {(trainingInfo.completed_games / trainingInfo.num_games) * 100}%"
					></div>
				</div>
			{/if}
		</div>
	{/if}
</main>
