import { findValidPlacementIndex } from './math';

const API_BASE = 'http://127.0.0.1:8080/api';

export class EngineState {
	gameState = $state<any>(null);
	selectedSlot = $state(-1);
	hoveredIdx = $state(-1);
	isTraining = $state(false);
	spectatorInterval = $state<ReturnType<typeof setInterval> | null>(null);
	loading = $state(true);

	isLeaderboardOpen = $state(false);
	topGames = $state<any[]>([]);
	vaultSortKey = $state('score');
	vaultSortDesc = $state(true);

	isReplaying = $state(false);
	replaySpeedMs = $state(400);
	replayInterval = $state<ReturnType<typeof setInterval> | null>(null);
	replayTimeoutId = $state<ReturnType<typeof setTimeout> | null>(null);
	spectatorLastReplayedId = $state(-1);
	replayStats = $state<any>(null);

	trainingInfo = $state<any>(null);
	trainingStatusInterval = $state<ReturnType<typeof setInterval> | null>(null);

	get sortedTopGames() {
		return [...this.topGames].sort((a, b) => {
			let valA = a[this.vaultSortKey];
			let valB = b[this.vaultSortKey];
			if (this.vaultSortDesc) return valB > valA ? 1 : valB < valA ? -1 : 0;
			return valA > valB ? 1 : valA < valB ? -1 : 0;
		});
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
			const res = await fetch(`${API_BASE}/games/top`);
			if (res.ok) {
				this.topGames = await res.json();
			}
		} catch (e) {
			console.error('Failed to fetch top games');
		}
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

	startSpectatorPolling() {
		if (this.spectatorInterval) return;
		this.spectatorInterval = setInterval(async () => {
			await this.fetchTopGames();
			if (this.topGames.length > 0) {
				const topGame = this.topGames[0];
				if (topGame.id !== this.spectatorLastReplayedId) {
					this.spectatorLastReplayedId = topGame.id;
					this.replayGame(topGame.id, true);
				} else if (!this.isReplaying) {
					this.replayGame(topGame.id, true);
				}
			}
		}, 3000);
	}

	async checkTrainingStatus() {
		try {
			const res = await fetch(`${API_BASE}/training/status`);
			if (res.ok) {
				const data = await res.json();
				this.trainingInfo = data;
				if (data.running && !this.isTraining) {
					this.isTraining = true;
					this.startSpectatorPolling();
				} else if (!data.running && this.isTraining) {
					this.isTraining = false;
					if (this.spectatorInterval) {
						clearInterval(this.spectatorInterval);
						this.spectatorInterval = null;
					}
					this.stopReplay();
				}
			}
		} catch (e) {}
	}

	async toggleTraining() {
		if (this.isTraining) {
			this.isTraining = false;
			if (this.spectatorInterval) {
				clearInterval(this.spectatorInterval);
				this.spectatorInterval = null;
			}
			this.stopReplay();
			this.spectatorLastReplayedId = -1;
			await fetch(`${API_BASE}/training/stop`, { method: 'POST' });
			this.fetchState();
		} else {
			this.isTraining = true;
			await fetch(`${API_BASE}/training/start`, { method: 'POST' });
			this.startSpectatorPolling();
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
		this.checkTrainingStatus();
		this.trainingStatusInterval = setInterval(() => this.checkTrainingStatus(), 1500);
	}

	unmount() {
		if (this.spectatorInterval) clearInterval(this.spectatorInterval);
		if (this.trainingStatusInterval) clearInterval(this.trainingStatusInterval);
	}
}

export const engine = new EngineState();
