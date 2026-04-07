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
    checkpoint_interval: 100,
    discount_factor: 0.99,
    td_lambda: 0.95,
    weight_decay: 0.0001,
  });

  const parameterGroups: GroupDef[] = [
    {
      title: "1. Neural Architecture & Topology",
      color: "text-purple-400",
      icon: Network,
      fields: [
        {
          key: "num_blocks",
          label: "ResNet Blocks",
          min: 2,
          max: 30,
          step: 1,
          tooltip:
            "Number of residual blocks spanning the deep neural network.",
        },
        {
          key: "hidden_dimension_size",
          label: "ResNet Channels",
          min: 32,
          max: 512,
          step: 32,
          tooltip: "Number of hidden dimension channels defining model width.",
        },
      ],
    },
    {
      title: "2. MDP & Value Estimation",
      color: "text-emerald-400",
      icon: Brain,
      fields: [
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

  const handlePresetChange = (level: number) => {
    setPresetLevel(level);
    const newConfig = { ...config };
    switch (level) {
      case 1: // Tiny Test
        newConfig.num_processes = 2;
        newConfig.train_batch_size = 128;
        newConfig.hidden_dimension_size = 32;
        newConfig.num_blocks = 2;
        newConfig.simulations = 50;
        break;
      case 2: // Low-end GPU (Laptop)
        newConfig.num_processes = 4;
        newConfig.train_batch_size = 512;
        newConfig.hidden_dimension_size = 64;
        newConfig.num_blocks = 4;
        newConfig.simulations = 200;
        break;
      case 3: // Mid-range GPU
        newConfig.num_processes = 8;
        newConfig.train_batch_size = 1024;
        newConfig.hidden_dimension_size = 128;
        newConfig.num_blocks = 10;
        newConfig.simulations = 800;
        break;
      case 4: // High-End Domestic GPU (e.g. RTX 4080)
        newConfig.num_processes = 16;
        newConfig.train_batch_size = 2048;
        newConfig.hidden_dimension_size = 256;
        newConfig.num_blocks = 15;
        newConfig.simulations = 1200;
        break;
      case 5: // Enthusiast / Multi-GPU (e.g. RTX 4090)
        newConfig.num_processes = 32;
        newConfig.train_batch_size = 4096;
        newConfig.hidden_dimension_size = 512;
        newConfig.num_blocks = 20;
        newConfig.simulations = 2000;
        break;
    }
    setConfig(newConfig);
  };

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
        Object.assign(baseConfig, config);

        await invoke("update_run_config", {
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
              groups={parameterGroups}
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
