import { useAppStore } from "@/store/useAppStore";
import { invoke as tauriInvoke } from "@tauri-apps/api/core";
import { PALETTE } from "@/store/useAppStore";
import type { Run } from "@/bindings/Run";

const isTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const invoke = async <T>(cmd: string, args?: Record<string, any>): Promise<T> => {
  if (isTauri) return tauriInvoke<T>(cmd, args);
  if (cmd === "list_runs") return [] as T;
  return null as T;
};

export function useAppActions() {
  const loadRuns = useAppStore((state) => state.loadRuns);
  const runToRename = useAppStore((state) => state.runToRename);
  const newName = useAppStore((state) => state.newName);
  const setRunToRename = useAppStore((state) => state.setRunToRename);
  const runToDelete = useAppStore((state) => state.runToDelete);

  const runToFlush = useAppStore((state) => state.runToFlush);
  const setRunToFlush = useAppStore((state) => state.setRunToFlush);

  const handleRename = async () => {
    if (!runToRename || !newName.trim()) return;
    try {
      await invoke("rename_run", { id: runToRename, newName });
      setRunToRename(null);
      await loadRuns();
    } catch (e) {
      console.error(e);
    }
  };

  const handleDelete = async () => {
    if (!runToDelete) return;
    try {
      await invoke("delete_run", { id: runToDelete });
      useAppStore.setState((state) => ({
        runToDelete: null,
        selectedRunId: state.selectedRunId === runToDelete ? null : state.selectedRunId,
        selectedDashboardRuns: state.selectedDashboardRuns.filter((id) => id !== runToDelete),
      }));
      await loadRuns();
    } catch (e) {
      console.error(e);
    }
  };

  const handleFlush = async () => {
    if (!runToFlush) return;
    try {
      await invoke("flush_run", { id: runToFlush });
      setRunToFlush(null);
      await loadRuns();
    } catch (e) {
      console.error(e);
    }
  };

  const handleEngineCmd = async (runId: string, cmd: string, force?: boolean) => {
    try {
      if (cmd === "start") {
        useAppStore.setState((state) => {
          const runs = state.selectedDashboardRuns.includes(runId)
            ? state.selectedDashboardRuns
            : [...state.selectedDashboardRuns, runId];

          const newColors = { ...state.runColors };
          if (!newColors[runId]) {
            newColors[runId] = PALETTE[Object.keys(newColors).length % PALETTE.length];
          }

          return {
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
  };

  const handleClone = async (run: Run) => {
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
  };

  return {
    handleRename,
    handleDelete,
    handleFlush,
    handleEngineCmd,
    handleClone,
  };
}
