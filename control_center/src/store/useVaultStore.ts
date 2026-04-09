import { create } from "zustand";
import { invoke as tauriInvoke } from "@tauri-apps/api/core";

const isTauri =
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const invoke = async <T>(
  cmd: string,
  args?: Record<string, any>,
): Promise<T> => {
  if (isTauri) {
    return tauriInvoke<T>(cmd, args);
  }
  return null as T;
};

export type SortBy = "score" | "lines" | "depth" | "time";
export type SortDirection = "asc" | "desc";

export interface FrontendGameStep {
  board_low: string;
  board_high: string;
  available: number[];
  action_taken: number;
  piece_identifier: number;
}

export interface FrontendVaultGame {
  source_run_id: string;
  source_run_name: string;
  run_type: string;
  difficulty_setting: number;
  episode_score: number;
  steps: FrontendGameStep[];
  lines_cleared: number;
  mcts_depth_mean: number;
  mcts_search_time_mean: number;
}

interface VaultStore {
  games: FrontendVaultGame[];
  loading: boolean;
  error: string | null;
  sortBy: SortBy;
  sortDirection: SortDirection;
  selectedGameIndex: number | null;

  setSortBy: (sortBy: SortBy) => void;
  toggleSortDirection: () => void;
  setSelectedGameIndex: (index: number | null) => void;
  fetchVault: () => Promise<void>;
}

export const useVaultStore = create<VaultStore>()((set, get) => ({
  games: [],
  loading: false,
  error: null,
  sortBy: "score",
  sortDirection: "desc",
  selectedGameIndex: null,

  setSortBy: (sortBy) => {
    const current = get().sortBy;
    if (current === sortBy) {
      get().toggleSortDirection();
    } else {
      set({ sortBy, sortDirection: "desc" });
    }
  },

  toggleSortDirection: () => {
    set({ sortDirection: get().sortDirection === "asc" ? "desc" : "asc" });
  },

  setSelectedGameIndex: (index) => set({ selectedGameIndex: index }),

  fetchVault: async () => {
    set({ loading: true, error: null, selectedGameIndex: null });
    try {
      const data = await invoke<FrontendVaultGame[]>("get_vault_games");
      set({ games: data });
    } catch (e: any) {
      set({ error: e.toString(), games: [] });
    } finally {
      set({ loading: false });
    }
  },
}));
