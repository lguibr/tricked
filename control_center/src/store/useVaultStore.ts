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

interface VaultStore {
  games: any[];
  loading: boolean;
  error: string | null;
  sortBy: SortBy;
  sortDirection: SortDirection;

  setSortBy: (sortBy: SortBy) => void;
  toggleSortDirection: () => void;

  fetchVault: (runId: string | null) => Promise<void>;
}

export const useVaultStore = create<VaultStore>()((set, get) => ({
  games: [],
  loading: false,
  error: null,
  sortBy: "score",
  sortDirection: "desc",

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

  fetchVault: async (runId) => {
    if (!runId) {
      set({ games: [], error: null });
      return;
    }
    set({ loading: true, error: null });
    try {
      const data = await invoke<any[]>("get_vault_games", { runId });
      set({ games: data });
    } catch (e: any) {
      set({ error: e.toString(), games: [] });
    } finally {
      set({ loading: false });
    }
  },
}));
