import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { useEngineStore } from './useEngineStore';

let mockConnections: any[] = [];

class MockWebSocket {
    onopen: any = null;
    onmessage: any = null;
    onclose: any = null;
    close = vi.fn();
    constructor(url: string) {
        mockConnections.push(this);
    }
}

describe('useEngineStore', () => {
    beforeEach(() => {
        mockConnections = [];
        useEngineStore.setState({
            gameState: null,
            trainingInfo: null,
            isTraining: false,
            isReplaying: false,
            replayCursor: 0,
            activeReplayData: null,
            wsReconnectAttempts: 0,
        });
        vi.useFakeTimers();
        vi.stubGlobal('WebSocket', MockWebSocket);
    });

    afterEach(() => {
        vi.useRealTimers();
        vi.unstubAllGlobals();
        vi.restoreAllMocks();
    });

    it('locks live telemetry updates when in replay mode', () => {
        useEngineStore.setState({ isReplaying: true, gameState: { score: 10 } });

        useEngineStore.getState().connectWebSocket();

        const wsInstance = mockConnections[0];
        const mockEvent = { data: JSON.stringify({ spectator: { score: 9000 } }) };

        // This should be ignored because isReplaying = true
        wsInstance.onmessage(mockEvent);
        expect(useEngineStore.getState().gameState.score).toBe(10);

        // Disable replay mode and verify it updates
        useEngineStore.setState({ isReplaying: false });
        wsInstance.onmessage(mockEvent);
        expect(useEngineStore.getState().gameState.score).toBe(9000);
    });

    it('implements exponential backoff on WS disconnect', () => {
        useEngineStore.getState().connectWebSocket();
        const wsInstance = mockConnections[0];

        expect(useEngineStore.getState().wsReconnectAttempts).toBe(0);

        // First disconnect
        wsInstance.onclose();
        expect(useEngineStore.getState().wsReconnectAttempts).toBe(1);

        // Fast forward 1000ms (2^0 * 1000)
        vi.advanceTimersByTime(1000);
        expect(mockConnections.length).toBe(2);

        const wsInstance2 = mockConnections[1];
        wsInstance2.onclose();
        expect(useEngineStore.getState().wsReconnectAttempts).toBe(2);

        // Fast forward 2000ms (2^1 * 1000)
        vi.advanceTimersByTime(2000);
        expect(mockConnections.length).toBe(3);
    });
});
