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

  // Actions
  startPolling: () => void;
  refreshData: () => Promise<void>;
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

  refreshData: async () => {
    try {
      if (!get().gameState) {
        const specRes = await fetch('/api/spectator');
        if (specRes.ok) {
          const specData = await specRes.json();
          if (!specData.error) {
            set({ gameState: specData });
          }
        }
      }

      const statRes = await fetch('/api/training/status');
      if (statRes.ok) {
        const statData = await statRes.json();
        const currentInfo = get().trainingInfo;
        let gps = currentInfo?.games_per_second || 0;

        if (currentInfo && statData.games_played > currentInfo.games_played) {
          const diff = statData.games_played - currentInfo.games_played;
          // 30 second interval
          const instantGps = diff / 30;
          gps = (0.2 * instantGps) + (0.8 * gps);
        }

        set({
          trainingInfo: { ...statData, games_per_second: gps },
          isTraining: statData.running
        });
      }
    } catch (err) {
      console.error('Failed to poll telemetry data', err);
    }
  },

  startPolling: () => {
    // Only set up the interval once
    get().refreshData();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    if (!(window as any).pollingInterval) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).pollingInterval = setInterval(() => {
        get().refreshData();
      }, 30000);
    }
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
