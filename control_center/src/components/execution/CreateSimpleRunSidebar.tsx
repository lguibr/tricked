import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Field, FieldLabel, FieldSet, FieldGroup } from "@/components/ui/field";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ParameterForm, GroupDef } from "./ParameterForm";
import { AlertCircle, Cpu, Brain, Network } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";

interface Run {
  id: string;
  name: string;
  status: string;
  type: string;
  config: string;
  tag?: string;
}

export function CreateSimpleRunSidebar({ onClose }: { onClose: () => void }) {
  const loadRuns = useAppStore((state) => state.loadRuns);
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [presetLevel, setPresetLevel] = useState(3);

  const [config, setConfig] = useState<Record<string, any>>({
    num_processes: 8,
    train_batch_size: 1024,
    simulations: 800,
    max_gumbel_k: 16,
    lr_init: 0.02,
    num_blocks: 10,
    hidden_dimension_size: 128,
    value_support_size: 300,
    checkpoint_interval: 100,
    discount_factor: 0.99,
    td_lambda: 0.95,
    weight_decay: 0.0001,
    buffer_capacity_limit: 100000,
    unroll_steps: 5,
    temporal_difference_steps: 5,
  });

  const parameterGroups: GroupDef[] = [
    {
      title: "1. Neural Architecture & Topology",
      color: "text-purple-400",
      icon: Network,
      fields: [
        {
          key: "buffer_capacity_limit",
          label: "Buffer Capacity",
          min: 1000,
          max: 1000000,
          step: 1000,
          tooltip: "Maximum number of game states to store in memory.",
        },
        {
          key: "unroll_steps",
          label: "Unroll Steps",
          min: 1,
          max: 20,
          step: 1,
          tooltip: "Number of steps unrolled in the recurrent dynamics network.",
        },
        {
          key: "temporal_difference_steps",
          label: "TD Steps",
          min: 1,
          max: 20,
          step: 1,
          tooltip: "n-step return horizon for training value targets.",
        },
        {
          key: "discount_factor",
          label: "Discount Factor",
          min: 0.9,
          max: 0.999,
          step: 0.001,
          tooltip: "Discount factor (gamma) for future rewards.",
        },
        {
          key: "td_lambda",
          label: "TD Lambda",
          min: 0.5,
          max: 1.0,
          step: 0.01,
          tooltip: "Lambda for generalized advantage estimation (GAE).",
        },
      ],
    },
    {
      title: "3. Search Dynamics (MCTS & Gumbel)",
      color: "text-blue-400",
      icon: Brain,
      fields: [
        {
          key: "simulations",
          label: "Simulations",
          min: 10,
          max: 2000,
          step: 10,
          tooltip: "Number of MCTS simulations per turn.",
        },
        {
          key: "max_gumbel_k",
          label: "Gumbel K",
          min: 4,
          max: 64,
          step: 1,
          tooltip:
            "Subset of actions evaluated in the Gumbel exploration phase.",
        },
      ],
    },
    {
      title: "4. Optimization & Gradient Dynamics",
      color: "text-red-400",
      icon: Network,
      fields: [
        {
          key: "lr_init",
          label: "Learning Rate",
          min: 0.0001,
          max: 0.1,
          step: 0.0001,
          tooltip: "Initial step size for the neural network optimizer.",
        },
        {
          key: "weight_decay",
          label: "Weight Decay",
          min: 0.0,
          max: 0.1,
          step: 0.0001,
          tooltip: "L2 regularization penalty for the optimizer.",
        },
        {
          key: "train_batch_size",
          label: "Batch Size",
          min: 64,
          max: 4096,
          step: 64,
          tooltip: "Number of experiences processed in a single backward pass.",
        },
      ],
    },
    {
      title: "5. Systems Concurrency & Hardware Utilization",
      color: "text-amber-400",
      icon: Cpu,
      fields: [
        {
          key: "num_processes",
          label: "Worker Threads",
          min: 1,
          max: 128,
          step: 1,
          tooltip: "Number of parallel worker processes for data generation.",
        },
        {
          key: "checkpoint_interval",
          label: "Checkpoint Interval",
          min: 10,
          max: 1000,
          step: 10,
          tooltip: "Training steps between model checkpoint saves.",
        },
      ],
    },
  ];

  const [groupPresets, setGroupPresets] = useState<number[]>([3, 3, 3, 3, 3]);

  const maps = {
    blocks: [2, 4, 10, 15, 20],
    channels: [32, 64, 128, 256, 512],
    discount: [0.9, 0.95, 0.99, 0.995, 0.999],
    lambda: [0.8, 0.85, 0.95, 0.97, 0.99],
    sims: [50, 200, 800, 1200, 2000],
    gumbel: [8, 12, 16, 24, 32],
    batch: [128, 512, 1024, 2048, 4096],
    lr: [0.05, 0.03, 0.02, 0.005, 0.001],
    decay: [0.01, 0.005, 0.0001, 0.00005, 0.00001],
    workers: [2, 4, 8, 16, 32],
    check: [50, 100, 100, 200, 500],
  };

  const applyPresetToGroup = (
    conf: Record<string, any>,
    idx: number,
    level: number,
  ) => {
    const lIdx = level - 1;
    if (idx === 0) {
      conf.num_blocks = maps.blocks[lIdx];
      conf.hidden_dimension_size = maps.channels[lIdx];
    } else if (idx === 1) {
      conf.discount_factor = maps.discount[lIdx];
      conf.td_lambda = maps.lambda[lIdx];
    } else if (idx === 2) {
      conf.simulations = maps.sims[lIdx];
      conf.max_gumbel_k = maps.gumbel[lIdx];
    } else if (idx === 3) {
      conf.train_batch_size = maps.batch[lIdx];
      conf.lr_init = maps.lr[lIdx];
      conf.weight_decay = maps.decay[lIdx];
    } else if (idx === 4) {
      conf.num_processes = maps.workers[lIdx];
      conf.checkpoint_interval = maps.check[lIdx];
    }
  };

  const handleGroupPresetChange = (groupIndex: number, level: number) => {
    const prev = [...groupPresets];
    prev[groupIndex] = level;
    setGroupPresets(prev);

    const newConfig = { ...config };
    applyPresetToGroup(newConfig, groupIndex, level);
    setConfig(newConfig);

    // If all match, update global preset. Else, set global to 0 (custom)
    if (prev.every((p) => p === level)) {
      setPresetLevel(level);
    } else {
      setPresetLevel(0);
    }
  };

  const handlePresetChange = (level: number) => {
    setPresetLevel(level);
    setGroupPresets([level, level, level, level, level]);
    const newConfig = { ...config };
    for (let i = 0; i < 5; i++) {
      applyPresetToGroup(newConfig, i, level);
    }
    setConfig(newConfig);
  };

  const currentGroups = parameterGroups.map((g, idx) => ({
    ...g,
    presetLevel: groupPresets[idx],
  }));

  const handleCreate = async () => {
    if (!name.trim()) {
      setError("Run name is required.");
      return;
    }
    setError("");

    try {
      const createdRun = await invoke<Run>("create_run", {
        name,
        type: "SINGLE",
        preset: "default",
      });

      try {
                const baseConfig = JSON.parse(createdRun.config || "{}");
        
        // Deep merge overrides into the nested base config safely without unsetting everything else
        if (!baseConfig.hardware) baseConfig.hardware = {};
        if (!baseConfig.architecture) baseConfig.architecture = {};
        if (!baseConfig.optimizer) baseConfig.optimizer = {};
        if (!baseConfig.mcts) baseConfig.mcts = {};

        if (config.num_processes !== undefined) baseConfig.hardware.num_processes = config.num_processes;
        if (config.train_batch_size !== undefined) baseConfig.optimizer.train_batch_size = config.train_batch_size;
        if (config.simulations !== undefined) baseConfig.mcts.simulations = config.simulations;
        if (config.max_gumbel_k !== undefined) baseConfig.mcts.max_gumbel_k = config.max_gumbel_k;
        if (config.lr_init !== undefined) baseConfig.optimizer.lr_init = config.lr_init;
        if (config.num_blocks !== undefined) baseConfig.architecture.num_blocks = config.num_blocks;
        if (config.hidden_dimension_size !== undefined) {
          baseConfig.architecture.hidden_dimension_size = config.hidden_dimension_size;
          baseConfig.architecture.spatial_channel_count = config.hidden_dimension_size;
        }
        if (config.value_support_size !== undefined) {
          baseConfig.architecture.value_support_size = config.value_support_size;
          baseConfig.architecture.reward_support_size = config.value_support_size;
        }
        if (config.buffer_capacity_limit !== undefined) baseConfig.optimizer.buffer_capacity_limit = config.buffer_capacity_limit;
        if (config.unroll_steps !== undefined) baseConfig.optimizer.unroll_steps = config.unroll_steps;
        if (config.temporal_difference_steps !== undefined) baseConfig.optimizer.temporal_difference_steps = config.temporal_difference_steps;
        if (config.checkpoint_interval !== undefined) baseConfig.checkpoint_interval = config.checkpoint_interval;
        if (config.discount_factor !== undefined) baseConfig.optimizer.discount_factor = config.discount_factor;
        if (config.td_lambda !== undefined) baseConfig.optimizer.td_lambda = config.td_lambda;
        if (config.weight_decay !== undefined) baseConfig.optimizer.weight_decay = config.weight_decay;

        await invoke("save_config", {
          id: createdRun.id,
          config: JSON.stringify(baseConfig, null, 2),
        });
      } catch (err) {
        console.warn("Failed to parse or save base config", err);
      }

      onClose();
      setName("");
      loadRuns();
    } catch (e) {
      console.error(e);
      setError(String(e));
    }
  };

  return (
    <div className="flex flex-col h-full overflow-hidden bg-[#09090b]">
      <div className="px-5 py-3 border-b border-border/10 flex justify-between items-center bg-black/20">
        <h3 className="text-xs font-bold text-zinc-100 uppercase tracking-widest">
          New Simple Run
        </h3>
      </div>
      <ScrollArea className="flex-1 p-4">
        <FieldSet className="flex flex-col gap-4">
          <FieldGroup className="gap-2">
            <Field>
              <FieldLabel
                htmlFor="new-name"
                className="text-[10px] text-zinc-400"
              >
                Experiment Name <span className="text-red-500">*</span>
              </FieldLabel>
              <Input
                id="new-name"
                value={name}
                onChange={(e) => {
                  setName(e.target.value);
                  if (error) setError("");
                }}
                placeholder="e.g. baseline_v2"
                className={`bg-zinc-900 border-border/30 text-sm ${
                  error ? "border-red-500/50 focus-visible:ring-red-500/20" : ""
                }`}
              />
              {error && (
                <div className="flex items-center gap-1 mt-1 text-red-500 text-[10px]">
                  <AlertCircle className="w-3 h-3" />
                  <span>{error}</span>
                </div>
              )}
            </Field>

            <div className="flex flex-col gap-2 p-3 bg-zinc-900/50 border border-border/20 rounded-md">
              <div className="flex justify-between items-center">
                <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-400">
                  Smart Hardware Preset
                </span>
                <span className="text-[10px] font-mono text-zinc-500">
                  Level {presetLevel} / 5
                </span>
              </div>
              <div className="text-[10px] text-zinc-500 mb-1">
                {presetLevel === 1 && "Tiny testing configuration (fastest)."}
                {presetLevel === 2 && "Low-end Laptop GPU limits."}
                {presetLevel === 3 && "Standard RTX 3060 / 4060 capabilities."}
                {presetLevel === 4 && "High-end RTX 4080 capabilities."}
                {presetLevel === 5 && "Enthusiast RTX 4090 absolute peak."}
              </div>
              <input
                type="range"
                min="1"
                max="5"
                step="1"
                value={presetLevel}
                onChange={(e) => handlePresetChange(parseInt(e.target.value))}
                className="w-full accent-emerald-500"
              />
            </div>

            <div className="h-px bg-border/20 my-2" />

            <ParameterForm
              mode="single"
              value={config}
              onChange={setConfig}
              groups={currentGroups}
              onGroupPresetChange={handleGroupPresetChange}
            />
          </FieldGroup>
        </FieldSet>
      </ScrollArea>
      <div className="p-4 border-t border-border/20 flex gap-2 shrink-0 bg-black/20">
        <Button
          variant="outline"
          className="flex-1 text-xs h-8"
          onClick={onClose}
        >
          Cancel
        </Button>
        <Button className="flex-1 text-xs h-8" onClick={handleCreate}>
          Create Run
        </Button>
      </div>
    </div>
  );
}
