import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { EngineState } from './state.svelte';

const mockGames = [
	{ id: 1, score: 50, steps: 10, difficulty: 6, moves: ['0', { board: '1', score: 10, available: [0, -1, -1] }] },
	{ id: 2, score: 20, steps: 5, difficulty: 3, moves: ['0'] }
];

global.fetch = vi.fn((url: string, options: any) => {
	if (url.includes('/games/top')) return Promise.resolve({ ok: true, json: () => Promise.resolve(mockGames) });
	if (url.includes('/games/1')) return Promise.resolve({ ok: true, json: () => Promise.resolve(mockGames[0]) });
	if (url.includes('/state')) return Promise.resolve({ ok: true, json: () => Promise.resolve({ board: '0', available: [0, 1, 2], piece_masks: { '0': Array(96).fill('1') } }) });
	if (url.includes('/training/status')) {
		if (options?.method === 'POST') return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
		return Promise.resolve({ ok: true, json: () => Promise.resolve({ running: true, iteration: 1, total_iterations: 10 }) });
	}
	if (url.includes('/reset')) return Promise.resolve({ ok: true, json: () => Promise.resolve({ board: '0' }) });
	if (url.includes('/rotate')) return Promise.resolve({ ok: true, json: () => Promise.resolve({ board: '0' }) });
	if (url.includes('/move')) return Promise.resolve({ ok: true, json: () => Promise.resolve({ board: '1', available: [-1,1,2] }) });
	
	return Promise.resolve({ ok: false, json: () => Promise.resolve({}) });
}) as any;

describe('EngineState Orchestrator', () => {
	let engine: EngineState;

	beforeEach(() => {
		engine = new EngineState();
		vi.useFakeTimers();
	});

	afterEach(() => {
		vi.restoreAllMocks();
		engine.unmount();
	});

	it('Mounts and fetches API', async () => {
		engine.mount();
		await new Promise(process.nextTick); 
		expect(engine.topGames.length).toBe(2);
		expect(engine.gameState).toBeDefined();
		expect(engine.isTraining).toBe(true);
		expect(engine.trainingInfo.iteration).toBe(1);
	});

	it('Toggles Leaderboard', async () => {
		await engine.toggleLeaderboard();
		expect(engine.isLeaderboardOpen).toBe(true);
		await engine.toggleLeaderboard();
		expect(engine.isLeaderboardOpen).toBe(false);
	});

	it('Clicks empty board safely', async () => {
		engine.selectedSlot = 0;
		const masks = Array(96).fill('0');
		masks[5] = '32'; // 1 << 5
		engine.gameState = { board: '0', available: [0, 1, 2], piece_masks: { '0': masks } };
		engine.hoveredIdx = 5;
		
		expect(engine.activeMaskStr).toBe('32');
		
		engine.isTraining = false;
		await engine.handleClick(5);
		expect(engine.selectedSlot).toBe(-1); 
	});

	it('Handles API interactions', async () => {
		await engine.resetGame(1);
		engine.gameState = { available: [0, 1, 2] };
		engine.isTraining = false;
		await engine.rotateSlot(0);
		
		engine.isTraining = true;
		await engine.toggleTraining();
		expect(engine.isTraining).toBe(false);
		
		await engine.toggleTraining();
		expect(engine.isTraining).toBe(true);
	});

	it('Replays Game timeline', async () => {
		await engine.replayGame(1);
		expect(engine.isReplaying).toBe(true);
		
		// Fast forward through timeouts
		vi.advanceTimersByTime(2000);
		expect(engine.replayStats).toBeDefined();
		
		engine.stopReplay();
		expect(engine.isReplaying).toBe(false);
	});

	it('Calculates Vault Sorting', async () => {
		engine.topGames = mockGames;
		engine.vaultSortKey = 'score';
		engine.vaultSortDesc = true;
		expect(engine.sortedTopGames[0].id).toBe(1);
		
		engine.vaultSortDesc = false;
		expect(engine.sortedTopGames[0].id).toBe(2);
	});
});
