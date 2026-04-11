import { create } from "zustand";
import { invoke as tauriInvoke } from "@tauri-apps/api/core";
import { generateRunName } from "@/lib/nameGenerator";

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
  initialRefineConfig: Record<string, any> | null;
  setInitialRefineConfig: (val: Record<string, any> | null) => void;
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
    resnetBlocks: { min: 4, max: 20 },
    resnetChannels: { min: 64, max: 256 },
    num_processes: { min: 4, max: 32 },
    train_batch_size: { min: 512, max: 2048 },
    simulations: { min: 100, max: 1000 },
    max_gumbel_k: { min: 8, max: 32 },
    lr_init: { min: 0.001, max: 0.01 },
    discount_factor: { min: 0.9, max: 0.999 },
    td_lambda: { min: 0.9, max: 0.99 },
    weight_decay: { min: 0.0, max: 0.01 },
    device: "cuda:0",
    value_support_size: { min: 50, max: 600 },
    reward_support_size: 300,
    spatial_channel_count: 64,
    hole_predictor_dim: 64,
    buffer_capacity_limit: { min: 10000, max: 500000 },
    checkpoint_interval: 100,
    worker_device: "cpu",
    unroll_steps: { min: 2, max: 10 },
    temporal_difference_steps: { min: 2, max: 10 },
    inference_batch_size_limit: 64,
    inference_timeout_ms: 50,
    gumbel_scale: 0.5,
    temp_decay_steps: 100000,
    difficulty: 0,
    temp_boost: true,
    reanalyze_ratio: 0.0,
  },
  studyName: `study-${generateRunName()}`,
  setStudyName: (val) => set({ studyName: val }),
  initialRefineConfig: null,
  setInitialRefineConfig: (val) => set({ initialRefineConfig: val }),
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
        resnetBlocks: config.resnetBlocks?.max ?? config.resnetBlocks,
        resnetChannels: config.resnetChannels?.max ?? config.resnetChannels,
        value_support_size: config.value_support_size,
        buffer_capacity_limit: config.buffer_capacity_limit,
        unroll_steps: config.unroll_steps,
        temporal_difference_steps: config.temporal_difference_steps,
      };

      const id =
        get().studyName.trim() ||
        `tuning_study_${Math.floor(Date.now() / 1000)}`;
      const name = id;

      const baseConfig = {
        experiment_name_identifier: name,
        checkpoint_interval: config.checkpoint_interval,
        hardware: {
          device: config.device || "cuda:0",
          num_processes: config.num_processes?.max ?? config.num_processes,
          worker_device: config.worker_device,
          inference_batch_size_limit: config.inference_batch_size_limit,
          inference_timeout_ms: config.inference_timeout_ms,
        },
        architecture: {
          hidden_dimension_size:
            config.hidden_dimension_size?.max ?? config.hidden_dimension_size,
          num_blocks: config.num_blocks?.max ?? config.num_blocks,
          value_support_size:
            config.value_support_size?.max ?? config.value_support_size,
          reward_support_size: config.reward_support_size,
          spatial_channel_count: config.spatial_channel_count,
          hole_predictor_dim: config.hole_predictor_dim,
        },
        optimizer: {
          buffer_capacity_limit:
            config.buffer_capacity_limit?.max ?? config.buffer_capacity_limit,
          train_batch_size:
            config.train_batch_size?.max ?? config.train_batch_size,
          discount_factor:
            config.discount_factor?.max ?? config.discount_factor,
          td_lambda: config.td_lambda?.max ?? config.td_lambda,
          weight_decay: config.weight_decay?.max ?? config.weight_decay,
          lr_init: config.lr_init?.max ?? config.lr_init,
          unroll_steps: config.unroll_steps?.max ?? config.unroll_steps,
          temporal_difference_steps:
            config.temporal_difference_steps?.max ??
            config.temporal_difference_steps,
          reanalyze_ratio: config.reanalyze_ratio,
        },
        mcts: {
          simulations: config.simulations?.max ?? config.simulations,
          max_gumbel_k: config.max_gumbel_k?.max ?? config.max_gumbel_k,
          gumbel_scale: config.gumbel_scale,
        },
        environment: {
          difficulty: config.difficulty,
          temp_decay_steps: config.temp_decay_steps,
          temp_boost: config.temp_boost,
        },
      };

      await invoke("start_study", {
        id,
        name,
        trials: config.trials,
        maxSteps: config.maxSteps,
        timeout: config.timeout,
        resnetBlocks: config.resnetBlocks?.max ?? config.resnetBlocks,
        resnetChannels: config.resnetChannels?.max ?? config.resnetChannels,
        bounds,
        baseConfig,
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
