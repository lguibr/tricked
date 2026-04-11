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
  "#6366f1",
  "#d946ef",
  "#0ea5e9",
  "#22c55e",
  "#eab308",
  "#f43f5e",
  "#a855f7",
  "#14b8a6",
  "#8b5cf6",
  "#f97066",
  "#c026d3",
  "#0284c7",
  "#16a34a",
  "#ca8a04",
  "#e11d48",
  "#9333ea",
  "#0d9488",
  "#7c3aed",
  "#ea580c",
  "#65a30d",
  "#db2777",
  "#2563eb",
  "#059669",
  "#d97706",
  "#be123c",
  "#7e22ce",
  "#0f766e",
  "#6d28d9",
  "#c2410c",
  "#4d7c0f",
  "#9f1239",
  "#1d4ed8",
  "#047857",
  "#b45309",
  "#4c1d95",
  "#0e7490",
  "#a21caf",
  "#15803d",
  "#5b21b6",
  "#a16207",
  "#8bc34a",
  "#cddc39",
  "#ffc107",
  "#ff9800",
  "#ff5722",
  "#795548",
  "#9e9e9e",
  "#607d8b",
  "#00bcd4",
  "#009688",
  "#4caf50",
  "#8bc34a",
  "#cddc39",
  "#ffeb3b",
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
  runColors: Record<string, string>;
  isCreatingRun: boolean;
  viewMode: "runs" | "studies" | "playground" | "vault" | "evaluation";
  activeJobs: ActiveJob[];
  hasInitializedSelection: boolean;

  // Dialog state
  runToRename: string | null;
  newName: string;
  runToDelete: string | null;
  runToFlush: string | null;

  initialRunConfig: Record<string, any> | null;
  setInitialRunConfig: (val: Record<string, any> | null) => void;

  // Actions
  setRuns: (runs: Run[]) => void;
  setSelectedRunId: (id: string | null) => void;
  setSelectedDashboardRuns: (
    ids: string[] | ((prev: string[]) => string[]),
  ) => void;
  setRunColors: (
    colors:
      | Record<string, string>
      | ((prev: Record<string, string>) => Record<string, string>),
  ) => void;
  setIsCreatingRun: (isCreating: boolean) => void;
  isCreatingStudy: boolean;
  setIsCreatingStudy: (isCreating: boolean) => void;
  setViewMode: (
    mode: "runs" | "studies" | "playground" | "vault" | "evaluation",
  ) => void;
  setActiveJobs: (jobs: ActiveJob[]) => void;

  setRunToRename: (id: string | null) => void;
  setNewName: (name: string) => void;
  setRunToDelete: (id: string | null) => void;
  setRunToFlush: (id: string | null) => void;

  setHasInitializedSelection: (val: boolean) => void;

  toggleDashboardRun: (id: string, pressed: boolean) => void;
  loadRuns: () => Promise<void>;
}

export const useAppStore = create<AppState>()((set, get) => ({
  runs: [],
  selectedRunId: null,
  selectedDashboardRuns: [],
  runColors: {},
  isCreatingRun: false,
  isCreatingStudy: false,
  viewMode: "runs",
  activeJobs: [],

  runToRename: null,
  newName: "",
  runToDelete: null,
  runToFlush: null,

  initialRunConfig: null,
  setInitialRunConfig: (val) => set({ initialRunConfig: val }),

  setRuns: (runs) => set({ runs }),
  setSelectedRunId: (id) => set({ selectedRunId: id }),
  setSelectedDashboardRuns: (val) =>
    set((state) => ({
      selectedDashboardRuns:
        typeof val === "function" ? val(state.selectedDashboardRuns) : val,
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
  setHasInitializedSelection: (val: boolean) =>
    set({ hasInitializedSelection: val }),

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
}));
