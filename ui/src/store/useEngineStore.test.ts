import { describe, vi, beforeEach, afterEach } from 'vitest';
import { useEngineStore } from './useEngineStore';

let mockConnections: any[] = [];

class MockWebSocket {
    onopen: any = null;
    onmessage: any = null;
    onclose: any = null;
    close = vi.fn();
    constructor() {
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
        });
        vi.useFakeTimers();
        vi.stubGlobal('WebSocket', MockWebSocket);
    });

    afterEach(() => {
        vi.useRealTimers();
        vi.unstubAllGlobals();
        vi.restoreAllMocks();
    });

});
