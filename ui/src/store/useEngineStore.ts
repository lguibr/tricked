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
  playHumanMove: (slot: number, idx: number) => Promise<void>;
  playAiMove: () => Promise<void>;
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
          // CHANGED: Calculate GPS on the frontend
          const currentInfo = get().trainingInfo;
          let gps = currentInfo?.games_per_second || 0;

          if (currentInfo && data.status.games_played > currentInfo.games_played) {
            const diff = data.status.games_played - currentInfo.games_played;
            // Assuming 50ms tick rate (20 ticks per sec)
            const instantGps = diff * 20;
            gps = (0.2 * instantGps) + (0.8 * gps);
          }

          set({
            trainingInfo: { ...data.status, games_per_second: gps },
            isTraining: data.status.running
          });
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
    const response = await fetch('/api/training/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      const errorText = await response.text();
      alert(`Start Training Error: ${errorText}`);
      throw new Error(errorText);
    }

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

  playHumanMove: async (slot, idx) => {
    await fetch('/api/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ slot, idx }),
    });
  },

  playAiMove: async () => {
    await fetch('/api/play_ai', { method: 'POST' });
  },
}));
