import { create } from 'zustand';

interface EngineState {
  // Live Telemetry
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  gameState: any | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  trainingInfo: any | null;
  isTraining: boolean;

  // Replay State (The Vault)
  isReplaying: boolean;
  replayCursor: number; // Current step in the scrubber
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  activeReplayData: any | null;

  wsReconnectAttempts: number;

  // Actions
  connectWebSocket: () => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  startTraining: (config: any) => Promise<void>;
  pauseTraining: () => Promise<void>;
  resumeTraining: () => Promise<void>;
  loadReplay: (gameId: number) => Promise<void>;
  setReplayCursor: (cursor: number | ((c: number) => number)) => void;
  setReplaying: (replaying: boolean) => void;
}

export const useEngineStore = create<EngineState>((set, get) => ({
  gameState: null,
  trainingInfo: null,
  isTraining: false,
  isReplaying: false,
  replayCursor: 0,
  activeReplayData: null,
  wsReconnectAttempts: 0,

  connectWebSocket: () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    const ws = new WebSocket(wsUrl);
    ws.onopen = () => {
      set({ wsReconnectAttempts: 0 });
    };
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (!get().isReplaying && data.spectator) {
          set({ gameState: data.spectator });
        }
        if (data.status) {
          set({ trainingInfo: data.status, isTraining: data.status.running });
        }
      } catch (err) {
        console.error('Failed to parse WS data', err);
      }
    };
    ws.onclose = () => {
      const attempts = get().wsReconnectAttempts;
      const delay = Math.min(1000 * Math.pow(2, attempts), 30000);
      set({ wsReconnectAttempts: attempts + 1 });
      setTimeout(() => get().connectWebSocket(), delay);
    };
  },

  startTraining: async (config) => {
    await fetch('/api/training/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    set({ isTraining: true });
  },

  pauseTraining: async () => {
    await fetch('/api/training/pause', { method: 'POST' });
    set({ isTraining: false });
  },

  resumeTraining: async () => {
    await fetch('/api/training/resume', { method: 'POST' });
    set({ isTraining: true });
  },

  loadReplay: async (gameId) => {
    const res = await fetch(`/api/games/${gameId}`);
    const data = await res.json();
    set({ activeReplayData: data, isReplaying: true, replayCursor: 0 });
  },

  setReplayCursor: (cursor) => {
    if (typeof cursor === 'function') {
      set({ replayCursor: cursor(get().replayCursor) });
    } else {
      set({ replayCursor: cursor });
    }
  },

  setReplaying: (replaying) => {
    set({ isReplaying: replaying });
  },
}));
