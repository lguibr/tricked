import { describe, it, expect, vi } from 'vitest';
import { render, fireEvent } from '@testing-library/react';
import { MissionControl } from './MissionControl';
import { useEngineStore } from '../store/useEngineStore';
import React from 'react';

vi.mock('../components/BoardVisualizer', () => ({
    BoardVisualizer: ({ onPlayMove }: { onPlayMove: (idx: number) => void }) => (
        <button data-testid="mock-board" onClick={() => onPlayMove(0)}>Board</button>
    )
}));
vi.mock('../components/HeartbeatMonitor', () => ({ HeartbeatMonitor: () => <div /> }));
vi.mock('../components/OrchestratorControls', () => ({ OrchestratorControls: () => <div /> }));

vi.mock('../store/useEngineStore', () => ({
    useEngineStore: vi.fn(),
}));

describe('MissionControl', () => {
    it('syncs human interaction with active slot state and calls playHumanMove', () => {
        const playHumanMoveMock = vi.fn();
        const connectWebSocketMock = vi.fn();

        vi.mocked(useEngineStore).mockImplementation((selector: any) => {
            const state = {
                connectWebSocket: connectWebSocketMock,
                playHumanMove: playHumanMoveMock,
                playAiMove: vi.fn(),
                isTraining: true,
                gameState: {
                    available: [1, -1, 2],
                    piece_masks: null,
                }
            };
            return selector(state);
        });

        const { getAllByText, getByTestId } = render(<MissionControl />);

        expect(connectWebSocketMock).toHaveBeenCalled();
        expect(getAllByText('Empty').length).toBeGreaterThan(0);

        fireEvent.click(getByTestId('mock-board'));
        expect(playHumanMoveMock).toHaveBeenCalledWith(0, 0);
    });
});
