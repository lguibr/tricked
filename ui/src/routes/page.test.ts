import { describe, it, expect, vi } from 'vitest';
import { render, fireEvent } from '@testing-library/svelte';
import Page from './+page.svelte';
import ControlsBar from '$lib/components/ControlsBar.svelte';
import EpochTelemetry from '$lib/components/EpochTelemetry.svelte';
import MatrixBackground from '$lib/components/MatrixBackground.svelte';
import PieceBuffer from '$lib/components/PieceBuffer.svelte';
import TriangoBoard from '$lib/components/TriangoBoard.svelte';
import VaultDataGrid from '$lib/components/VaultDataGrid.svelte';
import { engine } from '$lib/state.svelte';

global.fetch = vi.fn((url: string) => {
	if (url.includes('/games/top')) return Promise.resolve({ ok: true, json: () => Promise.resolve([]) });
	if (url.includes('/training/status')) return Promise.resolve({ ok: true, json: () => Promise.resolve({ running: true, iteration: 1, total_iterations: 10 }) });
	return Promise.resolve({ ok: true, json: () => Promise.resolve({ board: '0', available: [-1, -1, -1], piece_masks: { '0': Array(96).fill('0') }, score: 10, pieces_left: 96, difficulty: 6, terminal: false }) });
}) as any;

describe('Tricked Component Ecosystem', () => {
	it('mounts the application shell correctly in loading state', () => {
		engine.loading = true;
		const { getByText } = render(Page);
		expect(getByText(/Booting Svelte Neural Interface/i)).toBeDefined();
	});

	it('mounts subcomponents without exploding', async () => {
		engine.loading = false;
		engine.isTraining = true;
		engine.trainingInfo = { iteration: 1, total_iterations: 10 };
		engine.gameState = { board: '0', available: [0, 1, 2], piece_masks: { '0': Array(96).fill('32') }, score: 10, pieces_left: 96, difficulty: 6, terminal: false };

		const { getByText: cText } = render(ControlsBar);
		await fireEvent.click(cText(/1 - Easy/));
		await fireEvent.click(cText(/3 - Normal/));
		await fireEvent.click(cText(/6 - Master/));
		engine.isTraining = false;
		await fireEvent.click(cText(/Vault/));
		await fireEvent.click(cText(/START CORE/));

		render(EpochTelemetry);

		const mock32 = Array(32).fill({ id: 1, score: 50, steps: 10, difficulty: 6, moves: ['32', { score: 10, board: '32', available: [-1, -1, -1] }] });
		render(MatrixBackground, { props: { topGames: mock32 } });

		engine.gameState.piece_masks = { '0': Array(96).fill('32'), '1': Array(96).fill('32'), '2': Array(96).fill('32') };
		const { container: pCont } = render(PieceBuffer);
		// Click on piece buffers randomly
		if (pCont.firstChild) {
			await fireEvent.click(pCont.firstChild);
			await fireEvent.contextMenu(pCont.firstChild);
		}

		const { container: tCont } = render(TriangoBoard);

		engine.isLeaderboardOpen = true;
		engine.topGames = [{ id: 1, score: 50, steps: 10, difficulty: 6 }];
		const { getByText: vText, getAllByText: vAll } = render(VaultDataGrid);
		await fireEvent.click(vText(/Eff\./));

		engine.isReplaying = true;
		engine.replayStats = { currentStep: 1, maxStep: 10, gameId: 1, maxScore: 50 };
		render(TriangoBoard);
	});
});
