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

interface TuningStore {
  config: Record<string, any>;
  studyName: string;
  setStudyName: (val: string) => void;
  setConfig: (
    config:
      | Record<string, any>
      | ((prev: Record<string, any>) => Record<string, any>),
  ) => void;
  isActive: boolean;
  tuneComplete: boolean;
  refreshStatus: () => Promise<void>;
  checkActive: () => Promise<void>;
  startScan: () => Promise<void>;
  stopScan: () => Promise<void>;
  flushStudy: () => Promise<void>;
}

export const useTuningStore = create<TuningStore>((set, get) => ({
  config: {
    trials: 50,
    timeout: 1800,
    maxSteps: 50,
    resnetBlocks: 4,
    resnetChannels: 64,
    num_processes: { min: 1, max: 32 },
    train_batch_size: { min: 64, max: 2048 },
    simulations: { min: 10, max: 1000 },
    max_gumbel_k: { min: 4, max: 32 },
    lr_init: { min: 0.005, max: 0.1 },
    discount_factor: { min: 0.9, max: 0.999 },
    td_lambda: { min: 0.5, max: 1.0 },
    weight_decay: { min: 0.0, max: 0.1 },
  },
  studyName: `tuning_study_${Math.floor(Date.now() / 1000)}`,
  setStudyName: (val) => set({ studyName: val }),
  setConfig: (val) =>
    set((state) => ({
      config: typeof val === "function" ? val(state.config) : val,
    })),
  isActive: false,
  tuneComplete: false,

  refreshStatus: async () => {
    try {
      const status = await invoke<boolean>("get_study_status", {
        studyId: "unified_tune",
      });
      set({ tuneComplete: status });
    } catch (e) {
      console.error(e);
    }
  },

  checkActive: async () => {
    try {
      const active = await invoke<boolean>("get_active_study", {
        studyId: "unified_tune",
      });
      set({ isActive: active });
      await get().refreshStatus();
    } catch (e) {
      console.error(e);
    }
  },

  startScan: async () => {
    const { config, refreshStatus } = get();
    try {
      const bounds: Record<string, any> = {
        num_processes: config.num_processes,
        train_batch_size: config.train_batch_size,
        simulations: config.simulations,
        max_gumbel_k: config.max_gumbel_k,
        lr_init: config.lr_init,
        discount_factor: config.discount_factor,
        td_lambda: config.td_lambda,
        weight_decay: config.weight_decay,
      };

      const id =
        get().studyName.trim() ||
        `tuning_study_${Math.floor(Date.now() / 1000)}`;
      const name = id;

      await invoke("start_study", {
        id,
        name,
        trials: config.trials,
        maxSteps: config.maxSteps,
        timeout: config.timeout,
        resnetBlocks: config.resnetBlocks,
        resnetChannels: config.resnetChannels,
        bounds,
      });

      await refreshStatus();
      set({ isActive: true });
    } catch (e) {
      console.error(e);
      alert(e);
    }
  },

  stopScan: async () => {
    try {
      await invoke("stop_study", { force: false });
      set({ isActive: false });
    } catch (e) {
      console.error(e);
    }
  },

  flushStudy: async () => {
    try {
      if (!confirm(`Are you sure you want to wipe the global study database?`))
        return;
      await invoke("flush_study", { studyType: "UNIFIED" });
      await get().refreshStatus();
    } catch (e) {
      console.error(e);
      alert(e);
    }
  },
}));
