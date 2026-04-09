import { create } from "zustand";
import type { Run } from "@/bindings/Run";
import type { ActiveJob } from "@/bindings/ActiveJob";
import { invoke as tauriInvoke } from "@tauri-apps/api/core";

export const PALETTE = [
  "#3b82f6",
  "#ef4444",
  "#10b981",
  "#f59e0b",
  "#8b5cf6",
  "#ec4899",
  "#06b6d4",
  "#84cc16",
  "#f97316",
  "#14b8a6",
];

const isTauri =
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const invoke = async <T>(
  cmd: string,
  args?: Record<string, any>,
): Promise<T> => {
  if (isTauri) {
    return tauriInvoke<T>(cmd, args);
  }
  console.warn("Mocking Tauri invoke for browser env:", cmd, args);
  if (cmd === "list_runs") return [] as T;
  return null as T;
};

interface AppState {
  runs: Run[];
  selectedRunId: string | null;
  selectedDashboardRuns: string[];
  runLogs: Record<string, string[]>;
  runColors: Record<string, string>;
  isCreatingRun: boolean;
  viewMode: "runs" | "studies" | "playground" | "vault";
  activeJobs: ActiveJob[];
  hasInitializedSelection: boolean;

  // Dialog state
  runToRename: string | null;
  newName: string;
  runToDelete: string | null;
  runToFlush: string | null;

  // Actions
  setRuns: (runs: Run[]) => void;
  setSelectedRunId: (id: string | null) => void;
  setSelectedDashboardRuns: (
    ids: string[] | ((prev: string[]) => string[]),
  ) => void;
  setRunLogs: (
    logs:
      | Record<string, string[]>
      | ((prev: Record<string, string[]>) => Record<string, string[]>),
  ) => void;
  setRunColors: (
    colors:
      | Record<string, string>
      | ((prev: Record<string, string>) => Record<string, string>),
  ) => void;
  setIsCreatingRun: (isCreating: boolean) => void;
  isCreatingStudy: boolean;
  setIsCreatingStudy: (isCreating: boolean) => void;
  setViewMode: (mode: "runs" | "studies" | "playground" | "vault") => void;
  setActiveJobs: (jobs: ActiveJob[]) => void;

  setRunToRename: (id: string | null) => void;
  setNewName: (name: string) => void;
  setRunToDelete: (id: string | null) => void;
  setRunToFlush: (id: string | null) => void;

  setHasInitializedSelection: (val: boolean) => void;

  toggleDashboardRun: (id: string, pressed: boolean) => void;
  loadRuns: () => Promise<void>;
  handleRename: () => Promise<void>;
  handleDelete: () => Promise<void>;
  handleFlush: () => Promise<void>;
  handleEngineCmd: (
    runId: string,
    cmd: string,
    force?: boolean,
  ) => Promise<void>;
  handleClone: (run: Run) => Promise<void>;
}

export const useAppStore = create<AppState>()((set, get) => ({
  runs: [],
  selectedRunId: null,
  selectedDashboardRuns: [],
  runLogs: {},
  runColors: {},
  isCreatingRun: false,
  isCreatingStudy: false,
  viewMode: "runs",
  activeJobs: [],

  runToRename: null,
  newName: "",
  runToDelete: null,
  runToFlush: null,

  setRuns: (runs) => set({ runs }),
  setSelectedRunId: (id) => set({ selectedRunId: id }),
  setSelectedDashboardRuns: (val) =>
    set((state) => ({
      selectedDashboardRuns:
        typeof val === "function" ? val(state.selectedDashboardRuns) : val,
    })),
  setRunLogs: (val) =>
    set((state) => ({
      runLogs: typeof val === "function" ? val(state.runLogs) : val,
    })),
  setRunColors: (val) =>
    set((state) => ({
      runColors: typeof val === "function" ? val(state.runColors) : val,
    })),
  setIsCreatingRun: (isCreatingRun) => set({ isCreatingRun }),
  setIsCreatingStudy: (isCreatingStudy) => set({ isCreatingStudy }),
  setViewMode: (viewMode) => set({ viewMode }),
  setActiveJobs: (activeJobs) => set({ activeJobs }),

  setRunToRename: (runToRename) => set({ runToRename }),
  setNewName: (newName) => set({ newName }),
  setRunToDelete: (runToDelete) => set({ runToDelete }),
  setRunToFlush: (runToFlush) => set({ runToFlush }),

  toggleDashboardRun: (id, pressed) => {
    set((state) => {
      if (pressed)
        return { selectedDashboardRuns: [...state.selectedDashboardRuns, id] };
      return {
        selectedDashboardRuns: state.selectedDashboardRuns.filter(
          (r) => r !== id,
        ),
      };
    });
  },

  hasInitializedSelection: false,
  setHasInitializedSelection: (val: boolean) => set({ hasInitializedSelection: val }),

  loadRuns: async () => {
    try {
      const fetchedRuns = await invoke<Run[]>("list_runs");
      const currentColors = get().runColors;
      const previousRuns = get().runs;
      const currentSelected = get().selectedDashboardRuns;
      const hasInit = get().hasInitializedSelection;

      const newColors = { ...currentColors };
      let paletteIdx = Object.keys(newColors).length;
      let changedColors = false;

      const existingIds = new Set(previousRuns.map((r) => r.id));
      const newIds = fetchedRuns
        .filter((r) => !existingIds.has(r.id))
        .map((r) => r.id);

      fetchedRuns.forEach((r) => {
        if (!newColors[r.id]) {
          newColors[r.id] = PALETTE[paletteIdx % PALETTE.length];
          paletteIdx++;
          changedColors = true;
        }
      });

      let nextSelected = [...currentSelected];
      let nextHasInit = hasInit;
      let shouldUpdateSelection = false;

      if (!hasInit && fetchedRuns.length > 0) {
        nextSelected = fetchedRuns.map((r) => r.id);
        nextHasInit = true;
        shouldUpdateSelection = true;
        set({ selectedRunId: fetchedRuns[0].id });
      } else if (newIds.length > 0) {
        nextSelected = [...nextSelected, ...newIds];
        shouldUpdateSelection = true;
      }

      const updatePayload: Partial<AppState> = { runs: fetchedRuns };
      if (changedColors) updatePayload.runColors = newColors;
      if (shouldUpdateSelection) {
        updatePayload.selectedDashboardRuns = nextSelected;
        updatePayload.hasInitializedSelection = nextHasInit;
      }

      set(updatePayload as any);
    } catch (e) {
      console.error(e);
    }
  },

  handleRename: async () => {
    const { runToRename, newName, loadRuns } = get();
    if (!runToRename || !newName.trim()) return;
    try {
      await invoke("rename_run", { id: runToRename, newName });
      set({ runToRename: null });
      await loadRuns();
    } catch (e) {
      console.error(e);
    }
  },

  handleDelete: async () => {
    const { runToDelete, loadRuns } = get();
    if (!runToDelete) return;
    try {
      await invoke("delete_run", { id: runToDelete });
      set((state) => ({
        runToDelete: null,
        selectedRunId:
          state.selectedRunId === runToDelete ? null : state.selectedRunId,
        selectedDashboardRuns: state.selectedDashboardRuns.filter(
          (id) => id !== runToDelete,
        ),
      }));
      await loadRuns();
    } catch (e) {
      console.error(e);
    }
  },

  handleFlush: async () => {
    const { runToFlush, loadRuns } = get();
    if (!runToFlush) return;
    try {
      await invoke("flush_run", { id: runToFlush });
      set((state) => ({
        runToFlush: null,
        runLogs: { ...state.runLogs, [runToFlush]: [] },
      }));
      await loadRuns();
    } catch (e) {
      console.error(e);
    }
  },

  handleEngineCmd: async (runId, cmd, force) => {
    const { loadRuns } = get();
    try {
      if (cmd === "start") {
        set((state) => {
          const newLogs = { ...state.runLogs, [runId]: [] };
          const runs = state.selectedDashboardRuns.includes(runId)
            ? state.selectedDashboardRuns
            : [...state.selectedDashboardRuns, runId];

          const newColors = { ...state.runColors };
          if (!newColors[runId]) {
            newColors[runId] =
              PALETTE[Object.keys(newColors).length % PALETTE.length];
          }

          return {
            runLogs: newLogs,
            selectedDashboardRuns: runs,
            runColors: newColors,
          };
        });
        await invoke("start_run", { id: runId });
      } else if (cmd === "stop") {
        await invoke("stop_run", { id: runId, force });
      }
      await loadRuns();
    } catch (e) {
      console.error(e);
    }
  },

  handleClone: async (run) => {
    const { loadRuns } = get();
    try {
      const createdRun = await invoke<Run>("create_run", {
        name: run.name + "_clone",
        type: run.type,
        preset: "default",
      });
      await invoke("save_config", {
        id: createdRun.id,
        config: run.config,
      });
      await loadRuns();
    } catch (e) {
      console.error(e);
    }
  },
}));
