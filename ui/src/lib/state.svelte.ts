import { findValidPlacementIndex } from './math';
import { io, Socket } from 'socket.io-client';

const API_BASE = 'http://127.0.0.1:8080/api';

/**
 * High-performance UI Controller mapping native Svelte 5 `$state` mutations globally.
 * 
 * Dictates all interaction lifecycle limits, resolving API network translations and maintaining 
 * the real-time asynchronous telemetry pipeline representing underlying GPU operations.
 */
export class EngineState {
	socket: Socket | null = null;
	gameState = $state<any>(null);
	selectedSlot = $state(-1);
	hoveredIdx = $state(-1);
	isTraining = $state(false);
	loading = $state(true);

	// Tactical Hyperparameters
	tempDecaySteps = $state(30);
	maxGumbelK = $state(8);

	isLeaderboardOpen = $state(false);
	topGames = $state<any[]>([]);
	vaultSortKey = $state('score');
	vaultSortDesc = $state(true);
	vaultFilter = $state<number | null>(null);

	isReplaying = $state(false);
	spectatorLastReplayedId = $state(-1);
	replaySpeedMs = $state(400);
	replayInterval = $state<ReturnType<typeof setInterval> | null>(null);
	replayTimeoutId = $state<ReturnType<typeof setTimeout> | null>(null);
	replayStats = $state<any>(null);

	trainingInfo = $state<any>(null);
	topGamesInterval = $state<ReturnType<typeof setInterval> | null>(null);

	get sortedTopGames() {
		return [...this.topGames].filter(g => this.vaultFilter === null || g.difficulty === this.vaultFilter).sort((a, b) => {
			let valA = a[this.vaultSortKey];
			let valB = b[this.vaultSortKey];
			if (this.vaultSortDesc) return valB > valA ? 1 : valB < valA ? -1 : 0;
			return valA > valB ? 1 : valA < valB ? -1 : 0;
		});
	}

	get currentDifficulty() {
		return this.gameState ? this.gameState.difficulty : 1;
	}

	get totalEpochs() {
		return this.trainingInfo && this.trainingInfo.iteration !== undefined ? this.trainingInfo.iteration : 0;
	}

	get activeMaskStr() {
		if (this.selectedSlot === -1 || !this.gameState || this.hoveredIdx === -1) return null;
		const p_id = this.gameState.available[this.selectedSlot];
		if (p_id === -1) return null;
		const validIdx = findValidPlacementIndex(
			p_id,
			this.hoveredIdx,
			this.gameState.piece_masks,
			this.gameState.board
		);
		if (validIdx !== -1) {
			return this.gameState.piece_masks[p_id][validIdx];
		}
		return null;
	}

	async fetchTopGames() {
		try {
			const filterQuery = this.vaultFilter !== null ? `?difficulty=${this.vaultFilter}` : '';
			const res = await fetch(`${API_BASE}/games/top${filterQuery}`);
			if (res.ok) {
				this.topGames = await res.json();
			}
		} catch (e) {
			console.error('Failed to fetch top games');
		}
	}

	async fetchLeaderboard(filter: number | null) {
		this.vaultFilter = filter;
		await this.fetchTopGames();
	}

	async fetchState() {
		try {
			const res = await fetch(`${API_BASE}/state`);
			if (res.ok) this.gameState = await res.json();
		} catch (e) {
			console.error('Could not reach API');
		}
		this.loading = false;
	}

	async resetGame(diff = 6) {
		const res = await fetch(`${API_BASE}/reset`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ difficulty: diff })
		});
		if (res.ok) {
			this.gameState = await res.json();
			this.selectedSlot = -1;
		}
	}

	async rotateSlot(s: number) {
		if (!this.gameState || this.gameState.available[s] === -1 || this.isTraining) return;
		const res = await fetch(`${API_BASE}/rotate`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ slot: s })
		});
		if (res.ok) this.gameState = await res.json();
	}

	/**
	 * Natively evaluates physics collision predictions directly on the UI client via JSDoc masks 
	 * before executing a definitive POST mutation towards the Flask PyO3 interface.
	 * 
	 * @param anchorIdx - Extracted index where the mathematical layout initiates sequence operations.
	 */
	async handleClick(anchorIdx: number) {
		if (this.selectedSlot === -1 || !this.gameState || this.isTraining) return;
		const p_id = this.gameState.available[this.selectedSlot];
		if (p_id === -1) return;

		const validIdx = findValidPlacementIndex(
			p_id,
			anchorIdx,
			this.gameState.piece_masks,
			this.gameState.board
		);
		if (validIdx !== -1) {
			const res = await fetch(`${API_BASE}/move`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ slot: this.selectedSlot, idx: validIdx })
			});
			if (res.ok) {
				this.gameState = await res.json();
				this.selectedSlot = -1;
			}
		}
	}

	async toggleTraining() {
		if (this.isTraining) {
			this.isTraining = false;
			this.stopReplay();
			await fetch(`${API_BASE}/training/stop`, { method: 'POST' });
			this.fetchState();
		} else {
			this.isTraining = true;
			await fetch(`${API_BASE}/training/start`, { method: 'POST' });
		}
	}

	async toggleLeaderboard() {
		this.isLeaderboardOpen = !this.isLeaderboardOpen;
		if (this.isLeaderboardOpen) {
			await this.fetchTopGames();
		} else {
			this.stopReplay();
		}
	}

	stopReplay() {
		this.isReplaying = false;
		this.replayStats = null;
		if (this.replayInterval) {
			clearInterval(this.replayInterval);
			this.replayInterval = null;
		}
		if (this.replayTimeoutId) {
			clearTimeout(this.replayTimeoutId);
			this.replayTimeoutId = null;
		}
		this.fetchState();
	}

	async replayGame(id: number, isSpectatorLoop = false) {
		this.stopReplay();
		const res = await fetch(`${API_BASE}/games/${id}`);
		if (res.ok) {
			const data = await res.json();
			const moves = data.moves;
			this.isLeaderboardOpen = false;
			this.isReplaying = true;

			let step = 0;
			const maxStep = moves.length;
			if (this.gameState) {
				this.gameState.score = data.score;
				this.gameState.difficulty = data.difficulty;
				this.gameState.pieces_left = 96 - data.steps;
				this.gameState.terminal = true;
			}

			const playNextStep = () => {
				if (step < maxStep && this.gameState && this.isReplaying) {
					const moveData = moves[step];
					if (typeof moveData === 'string') {
						this.gameState.board = moveData;
					} else {
						this.gameState.board = moveData.board;
						this.gameState.score = moveData.score;
						this.gameState.available = moveData.available;
					}

					this.replayStats = {
						currentStep: step + 1,
						maxStep,
						gameId: id,
						maxScore: data.score
					};
					step++;
					this.replayTimeoutId = setTimeout(playNextStep, this.replaySpeedMs);
				} else {
					if (isSpectatorLoop && this.isTraining) {
						setTimeout(() => {
							if (this.isTraining && this.isReplaying) {
								this.replayGame(id, true);
							} else {
								this.stopReplay();
							}
						}, 2000);
					} else {
						this.stopReplay();
					}
				}
			};
			playNextStep();
		}
	}

	mount() {
		this.fetchState();
		this.fetchTopGames();

		fetch(`${API_BASE}/training/status`)
			.then(res => res.json())
			.then(data => {
				this.trainingInfo = data;
				if (data.running) this.isTraining = true;
			})
			.catch(() => { });

		this.socket = io('http://127.0.0.1:8080');

		this.socket.on('status', (data: any) => {
			this.trainingInfo = data;
			if (data.running && !this.isTraining) {
				this.isTraining = true;
			} else if (!data.running && this.isTraining) {
				this.isTraining = false;
				this.stopReplay();
				this.fetchState();
			}
		});

		this.socket.on('spectator', (data: any) => {
			if (this.isTraining && !this.isReplaying) {
				this.gameState = data;
			}
		});

		this.topGamesInterval = setInterval(() => this.fetchTopGames(), 5000);
	}

	unmount() {
		if (this.topGamesInterval) clearInterval(this.topGamesInterval);
		if (this.socket) {
			this.socket.disconnect();
			this.socket = null;
		}
	}
}

export const engine = new EngineState();
