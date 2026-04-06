import {
  Copy,
  Cpu,
  HardDrive,
  Layers,
  Brain,
  Clock,
  Activity,
  TrendingUp,
  Repeat,
  Zap,
  Users,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

const EXPLANATIONS: Record<string, string> = {
  simulations:
    "The number of MCTS simulations to run per step. Higher means deeper search but slower execution.",
  max_gumbel_k:
    "K parameter for Sequential Halving in Gumbel AlphaZero. Controls the number of considered top actions.",
  lr_init: "Initial learning rate for the neural network optimizer.",
  train_batch_size:
    "Number of transitions sampled from the replay buffer per training step.",
  num_processes:
    "Number of parallel worker processes generating self-play trajectories.",
  num_blocks: "Number of residual blocks in the core neural network.",
  device: "Primary compute device for the engine (e.g. cuda:0, cpu).",
  hidden_dimension_size:
    "The number of hidden channels in the neural network's convolution layers.",
  support_size:
    "The maximum absolute value supported by the discrete value distribution.",
  buffer_capacity_limit:
    "The maximum number of transitions to store in the replay buffer.",
  weight_decay: "L2 regularization penalty for the optimizer.",
  discount_factor: "Discount factor (gamma) for future rewards.",
  td_lambda: "Lambda for generalized advantage estimation (GAE).",
  checkpoint_interval: "Training steps between model checkpoint saves.",
  worker_device: "Compute device assigned to the rollout workers.",
  unroll_steps:
    "Number of steps into the future the network is trained to predict.",
  temporal_difference_steps: "TD-steps (n-step return) used for value targets.",
  inference_batch_size_limit:
    "Max batch size for parallel inference requests from workers.",
};

const getIcon = (key: string) => {
  const k = key.toLowerCase();
  if (k.includes("device")) return Cpu;
  if (k.includes("batch")) return Zap;
  if (k.includes("block") || k.includes("dimension") || k.includes("size"))
    return Layers;
  if (k.includes("lr")) return TrendingUp;
  if (k.includes("epoch") || k.includes("step")) return Repeat;
  if (k.includes("buffer") || k.includes("capacity")) return HardDrive;
  if (k.includes("mcts") || k.includes("gumbel") || k.includes("simulation"))
    return Brain;
  if (k.includes("time") || k.includes("timeout")) return Clock;
  if (k.includes("process") || k.includes("worker")) return Users;
  return Activity;
};

export function HydraConfigViewer({ configStr }: { configStr: string }) {
  let configObj: Record<string, any> = {};
  try {
    configObj = JSON.parse(configStr);
  } catch (e) {
    return <span className="text-red-400">Invalid config JSON.</span>;
  }

  const renderValue = (val: any) => {
    if (typeof val === "boolean") {
      return (
        <span className={val ? "text-emerald-400" : "text-rose-400"}>
          {val ? "true" : "false"}
        </span>
      );
    }
    if (typeof val === "number") {
      return <span className="text-sky-300">{val}</span>;
    }
    if (typeof val === "string") {
      return <span className="text-amber-200">"{val}"</span>;
    }
    if (Array.isArray(val)) {
      return <span className="text-fuchsia-300">[{val.join(", ")}]</span>;
    }
    return <span className="text-zinc-400">{JSON.stringify(val)}</span>;
  };

  const HARDWARE = [
    "device",
    "worker_device",
    "num_processes",
    "inference_batch_size_limit",
  ];
  const NETWORK = [
    "num_blocks",
    "hidden_dimension_size",
    "support_size",
    "checkpoint_interval",
    "checkpoint_history",
  ];
  const MCTS = ["simulations", "max_gumbel_k", "inference_timeout_ms"];
  const TRAINING = [
    "lr_init",
    "train_batch_size",
    "buffer_capacity_limit",
    "unroll_steps",
    "temporal_difference_steps",
    "discount_factor",
    "td_lambda",
    "weight_decay",
  ];

  const hardware: [string, any][] = [];
  const network: [string, any][] = [];
  const mcts: [string, any][] = [];
  const training: [string, any][] = [];
  const other: [string, any][] = [];

  Object.entries(configObj).forEach(([key, value]) => {
    if (HARDWARE.includes(key)) hardware.push([key, value]);
    else if (NETWORK.includes(key)) network.push([key, value]);
    else if (MCTS.includes(key)) mcts.push([key, value]);
    else if (TRAINING.includes(key)) training.push([key, value]);
    else other.push([key, value]);
  });

  const renderGroup = (title: string, items: [string, any][]) => {
    if (items.length === 0) return null;
    return (
      <div className="mb-3 last:mb-0">
        <h4 className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider mb-2 px-1 border-b border-border/20 pb-1">
          {title}
        </h4>
        <div className="flex flex-wrap gap-1.5">
          {items.map(([key, value]) => {
            const Icon = getIcon(key);
            const explanation =
              EXPLANATIONS[key] ||
              "Custom scaling parameter passed to the engine process overrides.";
            return (
              <Tooltip key={key} delayDuration={300}>
                <TooltipTrigger asChild>
                  <div className="flex items-center gap-1.5 px-2 py-1 bg-zinc-950/80 border border-border/40 rounded-md text-[10px] sm:text-xs font-mono cursor-help hover:bg-zinc-800 transition-colors shadow-sm">
                    <Icon className="w-3 h-3 text-emerald-500/80" />
                    <span className="text-zinc-500 font-medium truncate max-w-[120px]">
                      {key}
                    </span>
                    <span className="text-zinc-600">:</span>
                    <span className="font-bold truncate max-w-[120px]">
                      {renderValue(value)}
                    </span>
                  </div>
                </TooltipTrigger>
                <TooltipContent
                  side="bottom"
                  className="max-w-[250px] bg-zinc-900 border-border/50 text-xs text-zinc-200 shadow-xl"
                >
                  <p className="font-semibold text-emerald-400 mb-1">{key}</p>
                  <p className="text-zinc-400 leading-relaxed">{explanation}</p>
                </TooltipContent>
              </Tooltip>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="relative group/config mt-1 bg-[#0a0a0c]/50 border border-border/20 rounded-md p-3 flex flex-col gap-1 max-h-[300px] overflow-y-auto">
      <TooltipProvider>
        {renderGroup("Hardware & Distribution", hardware)}
        {renderGroup("MCTS Search Logic", mcts)}
        {renderGroup("Network Architecture", network)}
        {renderGroup("Training Hyperparameters", training)}
        {renderGroup("Other Setup", other)}
      </TooltipProvider>

      <Button
        variant="secondary"
        size="sm"
        className="absolute top-2 right-2 h-6 w-6 p-0 opacity-0 group-hover/config:opacity-100 transition-opacity bg-zinc-800 hover:bg-zinc-700 shadow-md"
        onClick={(e) => {
          e.stopPropagation();
          navigator.clipboard.writeText(JSON.stringify(configObj, null, 2));
        }}
      >
        <Copy className="w-3 h-3 text-zinc-300" />
      </Button>
    </div>
  );
}
