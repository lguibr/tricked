import { create } from "zustand";
import { invoke } from "@/lib/apiBridge";

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
  difficulty: number;
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
  flushVault: () => Promise<void>;
  emptyVault: () => Promise<void>;
  removeVaultGame: (game: FrontendVaultGame) => Promise<void>;
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

  flushVault: async () => {
    set({ loading: true, error: null });
    try {
      await invoke("flush_global_vault");
      const data = await invoke<FrontendVaultGame[]>("get_vault_games");
      set({ games: data, selectedGameIndex: null });
    } catch (e: any) {
      set({ error: e.toString() });
    } finally {
      set({ loading: false });
    }
  },

  emptyVault: async () => {
    set({ loading: true, error: null });
    try {
      await invoke("empty_all_vaults");
      const data = await invoke<FrontendVaultGame[]>("get_vault_games");
      set({ games: data, selectedGameIndex: null });
    } catch (e: any) {
      set({ error: e.toString() });
    } finally {
      set({ loading: false });
    }
  },

  removeVaultGame: async (game: FrontendVaultGame) => {
    set({ loading: true, error: null });
    try {
      await invoke("remove_vault_game", {
        score: game.episode_score,
        steps: game.steps.length,
        run_name: game.source_run_name,
      });
      // Refresh vault explicitly after removing
      const data = await invoke<FrontendVaultGame[]>("get_vault_games");
      set({ games: data, selectedGameIndex: null });
    } catch (e: any) {
      set({ error: e.toString() });
    } finally {
      set({ loading: false });
    }
  },
}));
