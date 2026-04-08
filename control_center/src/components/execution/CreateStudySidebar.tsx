import { useState } from "react";
import {
  VscPlay,
  VscSync,
  VscSettingsGear,
  VscTypeHierarchy,
  VscServerProcess,
  VscRepoForked,
  VscLightbulb,
  VscPulse,
} from "react-icons/vsc";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ParameterForm, GroupDef } from "./ParameterForm";
import { useTuningStore } from "@/store/useTuningStore";
import { useAppStore } from "@/store/useAppStore";
import { VscClose } from "react-icons/vsc";

export function CreateStudySidebar({ onClose }: { onClose: () => void }) {
  const config = useTuningStore((state) => state.config);
  const setConfig = useTuningStore((state) => state.setConfig);
  const isActive = useTuningStore((state) => state.isActive);
  const tuneComplete = useTuningStore((state) => state.tuneComplete);
  const startScan = useTuningStore((state) => state.startScan);

  const [presetLevel, setPresetLevel] = useState(3);
  const [groupPresets, setGroupPresets] = useState<number[]>([3, 3, 3, 3]);

  // Optuna limits differ slightly, we shift bounds dynamically based on level
  const applyPresetToGroup = (
    conf: Record<string, any>,
    idx: number,
    level: number,
  ) => {
    const lIdx = level - 1;
    if (idx === 0) {
      // Neural Architecture (Single values in tuning)
      const blocksMap = [2, 4, 10, 15, 20];
      const channelsMap = [32, 64, 128, 256, 512];
      conf.resnetBlocks = blocksMap[lIdx];
      conf.resnetChannels = channelsMap[lIdx];
    } else if (idx === 1) {
      // MDP & Value Estimation (Bounds)
      const maxDiscount = [0.95, 0.99, 0.999, 0.999, 0.999];
      const minDiscount = [0.9, 0.9, 0.9, 0.95, 0.98];
      conf.discount_factor = { min: minDiscount[lIdx], max: maxDiscount[lIdx] };
      const maxLambda = [0.9, 0.95, 0.99, 1.0, 1.0];
      const minLambda = [0.5, 0.8, 0.9, 0.95, 0.95];
      conf.td_lambda = { min: minLambda[lIdx], max: maxLambda[lIdx] };
    } else if (idx === 2) {
      // Search Dynamics (Bounds)
      const maxSims = [100, 400, 1000, 1500, 2000];
      const minSims = [10, 50, 100, 400, 800];
      conf.simulations = { min: minSims[lIdx], max: maxSims[lIdx] };
      const maxGumbel = [8, 16, 32, 48, 64];
      const minGumbel = [4, 4, 8, 16, 24];
      conf.max_gumbel_k = { min: minGumbel[lIdx], max: maxGumbel[lIdx] };
    } else if (idx === 3) {
      // Optimization (Bounds)
      const maxLr = [0.1, 0.05, 0.01, 0.005, 0.001];
      const minLr = [0.01, 0.005, 0.001, 0.0005, 0.0001];
      conf.lr_init = { min: minLr[lIdx], max: maxLr[lIdx] };

      const maxDecay = [0.1, 0.05, 0.01, 0.005, 0.001];
      const minDecay = [0.0, 0.0, 0.0, 0.0, 0.0];
      conf.weight_decay = { min: minDecay[lIdx], max: maxDecay[lIdx] };

      const maxBatch = [256, 1024, 2048, 4096, 4096];
      const minBatch = [64, 128, 512, 1024, 2048];
      conf.train_batch_size = { min: minBatch[lIdx], max: maxBatch[lIdx] };
    } else if (idx === 4) {
      // Systems (Bounds)
      const maxWorkers = [4, 8, 32, 64, 128];
      const minWorkers = [1, 2, 4, 16, 32];
      conf.num_processes = { min: minWorkers[lIdx], max: maxWorkers[lIdx] };
    }
  };

  const handleGroupPresetChange = (groupIndex: number, level: number) => {
    const prev = [...groupPresets];
    prev[groupIndex] = level;
    setGroupPresets(prev);

    const newConfig = { ...config };
    applyPresetToGroup(newConfig, groupIndex, level);
    setConfig(newConfig);

    if (prev.every((p) => p === level)) {
      setPresetLevel(level);
    } else {
      setPresetLevel(0);
    }
  };

  const handleGlobalPresetChange = (level: number) => {
    setPresetLevel(level);
    setGroupPresets([level, level, level, level, level]);
    const newConfig = { ...config };
    for (let i = 0; i < 5; i++) {
      applyPresetToGroup(newConfig, i, level);
    }
    setConfig(newConfig);
  };

  const singleGroups: GroupDef[] = [
    {
      title: "Optuna Global Controls",
      color: "text-zinc-300",
      icon: VscSettingsGear,
      fields: [
        {
          key: "trials",
          label: "Max Trials",
          min: 10,
          max: 1000,
          step: 10,
          tooltip: "Maximum number of tuning trials to perform.",
        },
        {
          key: "timeout",
          label: "Timeout (s)",
          min: 10,
          max: 7200,
          step: 60,
          tooltip:
            "Maximum time permitted for the study before early stopping.",
        },
        {
          key: "maxSteps",
          label: "Steps/Trial",
          min: 1,
          max: 100,
          step: 1,
          tooltip: "Maximum steps optimized in each individual trial.",
        },
      ],
    },
    {
      title: "1. Neural Architecture & Topology",
      color: "text-purple-400",
      icon: VscTypeHierarchy,
      presetLevel: groupPresets[0],
      fields: [
        {
          key: "resnetBlocks",
          label: "ResNet Blocks",
          min: 2,
          max: 30,
          step: 1,
          tooltip:
            "Number of residual blocks spanning the deep neural network.",
        },
        {
          key: "resnetChannels",
          label: "ResNet Channels",
          min: 32,
          max: 512,
          step: 32,
          tooltip: "Number of hidden dimension channels defining model width.",
        },
      ],
    },
  ];

  const boundGroups: GroupDef[] = [
    {
      title: "2. MDP & Value Estimation",
      color: "text-emerald-400",
      icon: VscLightbulb,
      presetLevel: groupPresets[1],
      fields: [
        {
          key: "discount_factor",
          label: "Discount Range",
          min: 0.9,
          max: 0.999,
          step: 0.001,
          tooltip: "Optuna will search this discount factor space.",
        },
        {
          key: "td_lambda",
          label: "TD Lambda Range",
          min: 0.5,
          max: 1.0,
          step: 0.01,
          tooltip: "Optuna will search this TD lambda space.",
        },
      ],
    },
    {
      title: "3. Search Dynamics",
      color: "text-blue-400",
      icon: VscRepoForked,
      presetLevel: groupPresets[2],
      fields: [
        {
          key: "simulations",
          label: "Simulations",
          min: 10,
          max: 2000,
          step: 10,
          tooltip: "Tree search explorations.",
        },
        {
          key: "max_gumbel_k",
          label: "Max Gumbel K",
          min: 4,
          max: 64,
          step: 1,
          tooltip: "Action subset size for Gumbel sampling.",
        },
      ],
    },
    {
      title: "4. Optimization & Gradient",
      color: "text-red-400",
      icon: VscPulse,
      presetLevel: groupPresets[3],
      fields: [
        {
          key: "lr_init",
          label: "Learning Rate",
          min: 0.0001,
          max: 0.1,
          step: 0.0001,
          tooltip: "Optuna will search this learning rate space.",
        },
        {
          key: "weight_decay",
          label: "Weight Decay Range",
          min: 0.0,
          max: 0.1,
          step: 0.0001,
          tooltip: "Optuna will search this L2 regularization space.",
        },
        {
          key: "train_batch_size",
          label: "Train Batch Size",
          min: 64,
          max: 4096,
          step: 64,
          tooltip:
            "Range for learning batch sizes sent through backpropagation.",
        },
      ],
    },
    {
      title: "5. Systems Concurrency",
      color: "text-amber-400",
      icon: VscServerProcess,
      presetLevel: groupPresets[4],
      fields: [
        {
          key: "num_processes",
          label: "Worker Processes",
          min: 1,
          max: 128,
          step: 1,
          tooltip: "Range for exploring data generation concurrency.",
        },
      ],
    },
  ];

  return (
    <div className="flex flex-col h-full bg-[#0a0a0a]">
      <div className="px-3 py-2 border-b border-border/10 flex justify-between items-center bg-black/20 shrink-0">
        <h3 className="text-xs font-bold text-zinc-100 uppercase tracking-widest flex items-center gap-1.5">
          <VscSettingsGear className="text-emerald-500" />
          New Tuning Study
        </h3>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6 text-zinc-400 hover:text-white hover:bg-white/10"
          onClick={onClose}
        >
          <VscClose size={14} />
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
        <div className="flex flex-col gap-3">
          <Card
            className={`bg-[#080808] border p-3 flex flex-col gap-3 shrink-0 transition-colors shadow-inner ${tuneComplete ? "border-emerald-500/30" : "border-white/5"}`}
          >
            <div className="p-4 border-b border-border/5 shrink-0 bg-[#080808]">
              <div className="flex items-center gap-2 mb-1">
                <VscSettingsGear className="w-4 h-4 text-purple-400" />
                <h2 className="text-[11px] font-black uppercase tracking-widest text-zinc-100">
                  Optuna Study Config
                </h2>
              </div>
              <p className="text-[10px] text-zinc-500 mb-4">
                Adjust hyperparameter search bounds. Save & initialize study
                parameters directly from this tab.
              </p>

              <div className="flex flex-col gap-2 mt-4">
                <label className="text-[9px] font-bold uppercase tracking-widest text-zinc-400 mb-1">
                  Study Name / Identifier
                </label>
                <input
                  type="text"
                  className="w-full bg-[#111] border border-white/10 rounded px-2 py-1 text-[11px] font-mono text-zinc-200 outline-none focus:border-purple-500/50"
                  placeholder="tuning_study_XYZ"
                  value={
                    useTuningStore((state: any) => state.studyName) || ""
                  }
                  onChange={(e) =>
                    useTuningStore.getState().setStudyName(e.target.value)
                  }
                />
              </div>
            </div>

            <p className="text-[9px] text-zinc-400 leading-tight uppercase font-mono tracking-tight pb-1">
              Multi-objective optimization mapping hardware throughput limits
              and training convergence potential simultaneously.
            </p>

            <div className="flex flex-col gap-3">
              {!isActive && (
                <div className="flex flex-col gap-2 p-3 bg-zinc-900/50 border border-border/20 rounded-md">
                  <div className="flex justify-between items-center">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-400">
                      Global Bounds Preset
                    </span>
                    <span className="text-[10px] font-mono text-zinc-500">
                      Level{" "}
                      {presetLevel === 0 ? "Custom" : `${presetLevel} / 5`}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="5"
                    step="1"
                    value={presetLevel || 3}
                    onChange={(e) =>
                      handleGlobalPresetChange(parseInt(e.target.value))
                    }
                    className="w-full accent-emerald-500"
                  />
                </div>
              )}

              {!isActive && (
                <div className="flex flex-col gap-3 pt-1">
                  <ParameterForm
                    mode="single"
                    value={config}
                    onChange={setConfig}
                    groups={singleGroups}
                    onGroupPresetChange={(idx, level) => {
                      // Note: singleGroups has Optuna Global Controls at idx 0, and Architecture at 1
                      // we only want to scale Architecture, which corresponds to our internal idx 0
                      if (idx === 1) handleGroupPresetChange(0, level);
                    }}
                  />
                  <ParameterForm
                    mode="bounds"
                    value={config}
                    onChange={setConfig}
                    groups={boundGroups}
                    onGroupPresetChange={(idx, level) => {
                      // boundsGroups are indices 1 to 4 mapping to 1 to 4 internally
                      handleGroupPresetChange(idx + 1, level);
                    }}
                  />
                </div>
              )}

              <div className="flex gap-1.5 mt-2">
                <Button
                  disabled={isActive}
                  onClick={async () => {
                    await startScan();
                    useAppStore.getState().loadRuns();
                    onClose();
                  }}
                  className="flex-1 bg-emerald-600/20 text-emerald-400 hover:bg-emerald-600/40 border border-emerald-500/40 shadow-none text-[9px] h-7 font-black tracking-widest uppercase"
                >
                  {isActive ? (
                    <VscSync className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                  ) : (
                    <VscPlay className="w-3.5 h-3.5 mr-1.5" />
                  )}
                  {isActive
                    ? "Starting..."
                    : "Initialize Search"}
                </Button>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
