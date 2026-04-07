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
  Check,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useState } from "react";

const EXPLANATIONS: Record<string, string> = {
  simulations: "The number of MCTS simulations to run per step. Higher means deeper search but slower execution.",
  max_gumbel_k: "K parameter for Sequential Halving in Gumbel AlphaZero. Controls the number of considered top actions.",
  lr_init: "Initial learning rate for the neural network optimizer.",
  train_batch_size: "Number of transitions sampled from the replay buffer per training step.",
  num_processes: "Number of parallel worker processes generating self-play trajectories.",
  num_blocks: "Number of residual blocks in the core neural network.",
  device: "Primary compute device for the engine (e.g. cuda:0, cpu).",
  hidden_dimension_size: "The number of hidden channels in the neural network's convolution layers.",
  support_size: "The maximum absolute value supported by the discrete value distribution.",
  buffer_capacity_limit: "The maximum number of transitions to store in the replay buffer.",
  weight_decay: "L2 regularization penalty for the optimizer.",
  discount_factor: "Discount factor (gamma) for future rewards.",
  td_lambda: "Lambda for generalized advantage estimation (GAE).",
  checkpoint_interval: "Training steps between model checkpoint saves.",
  worker_device: "Compute device assigned to the rollout workers.",
  unroll_steps: "Number of steps into the future the network is trained to predict.",
  temporal_difference_steps: "TD-steps (n-step return) used for value targets.",
  inference_batch_size_limit: "Max batch size for parallel inference requests from workers.",
};

const getIcon = (key: string) => {
  const k = key.toLowerCase();
  if (k.includes("device")) return Cpu;
  if (k.includes("batch")) return Zap;
  if (k.includes("block") || k.includes("dimension") || k.includes("size")) return Layers;
  if (k.includes("lr")) return TrendingUp;
  if (k.includes("epoch") || k.includes("step")) return Repeat;
  if (k.includes("buffer") || k.includes("capacity")) return HardDrive;
  if (k.includes("mcts") || k.includes("gumbel") || k.includes("simulation")) return Brain;
  if (k.includes("time") || k.includes("timeout")) return Clock;
  if (k.includes("process") || k.includes("worker")) return Users;
  return Activity;
};

// Categorization helper
const categorizeParams = (configObj: Record<string, any>) => {
  const HARDWARE = ["device", "worker_device", "num_processes", "inference_batch_size_limit", "checkpoint_interval", "checkpoint_history"];
  const NETWORK = ["num_blocks", "hidden_dimension_size", "support_size", "resnetBlocks", "resnetChannels"];
  const MCTS = ["simulations", "max_gumbel_k", "inference_timeout_ms"];
  const TRAINING = ["lr_init", "train_batch_size", "buffer_capacity_limit", "unroll_steps", "temporal_difference_steps", "discount_factor", "td_lambda", "weight_decay"];

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

  return { hardware, network, mcts, training, other };
};

// Visual themes for groups
const GROUP_THEMES: Record<string, { theme: string, iconColor: string, badgeBg: string, badgeBorder: string, labelText: string }> = {
  "Hardware & Distribution": { theme: "amber", iconColor: "text-amber-400", badgeBg: "bg-amber-500/10", badgeBorder: "border-amber-500/20", labelText: "text-amber-500/80" },
  "Network Architecture": { theme: "purple", iconColor: "text-purple-400", badgeBg: "bg-purple-500/10", badgeBorder: "border-purple-500/20", labelText: "text-purple-500/80" },
  "MCTS Search Logic": { theme: "blue", iconColor: "text-blue-400", badgeBg: "bg-blue-500/10", badgeBorder: "border-blue-500/20", labelText: "text-blue-500/80" },
  "Training Hyperparameters": { theme: "emerald", iconColor: "text-emerald-400", badgeBg: "bg-emerald-500/10", badgeBorder: "border-emerald-500/20", labelText: "text-emerald-500/80" },
  "Other Setup": { theme: "zinc", iconColor: "text-zinc-400", badgeBg: "bg-zinc-500/10", badgeBorder: "border-zinc-500/20", labelText: "text-zinc-500/80" },
};

// Common value renderer
const renderValue = (val: any) => {
  if (typeof val === "boolean") {
    return <span className={val ? "text-emerald-400" : "text-rose-400"}>{val ? "TRUE" : "FALSE"}</span>;
  }
  if (typeof val === "number") {
    return <span className="text-white font-extrabold tracking-tight">{val}</span>;
  }
  if (typeof val === "string") {
    return <span className="text-zinc-200">"{val}"</span>;
  }
  if (Array.isArray(val)) {
    return <span className="text-zinc-300">[{val.join(", ")}]</span>;
  }
  return <span className="text-zinc-400">{JSON.stringify(val)}</span>;
};

// -----------------------------------------------------------------------------
// Component logic
// -----------------------------------------------------------------------------

export function HydraConfigViewer({ configStr }: { configStr: string }) {
  const [copied, setCopied] = useState(false);

  let configObj: Record<string, any> = {};
  try {
    configObj = JSON.parse(configStr);
  } catch (e) {
    return <span className="text-red-400">Invalid config JSON.</span>;
  }

  const { hardware, network, mcts, training, other } = categorizeParams(configObj);

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(JSON.stringify(configObj, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const renderGroup = (title: string, items: [string, any][]) => {
    if (items.length === 0) return null;
    const theme = GROUP_THEMES[title] || GROUP_THEMES["Other Setup"];

    return (
      <div className="mb-4 last:mb-0">
        <h4 className={`text-[11px] font-black uppercase tracking-widest mb-2 px-1 border-b pb-1 flex items-center gap-1.5 ${theme.badgeBorder} ${theme.iconColor}`}>
          {title}
        </h4>
        <div className="flex flex-wrap gap-2">
          {items.map(([key, value]) => {
            const Icon = getIcon(key);
            const explanation = EXPLANATIONS[key] || "Custom scaling parameter passed to the engine process overrides.";
            return (
              <Tooltip key={key} delayDuration={300}>
                <TooltipTrigger asChild>
                  <div className={`flex items-center gap-2 px-2.5 py-1 ${theme.badgeBg} border ${theme.badgeBorder} rounded-md text-[10px] sm:text-xs font-mono cursor-help hover:brightness-125 transition-all shadow-sm`}>
                    <Icon className={`w-3.5 h-3.5 ${theme.iconColor}`} />
                    <span className={`${theme.labelText} font-semibold truncate max-w-[140px]`}>
                      {key}
                    </span>
                    <div className="w-[1px] h-3 bg-white/20 mx-0.5" />
                    <span className="font-bold truncate max-w-[140px] drop-shadow-md">
                      {renderValue(value)}
                    </span>
                  </div>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="max-w-[280px] bg-zinc-950 border-white/10 text-xs text-zinc-300 shadow-2xl p-3">
                  <p className={`font-black uppercase tracking-wider mb-1 ${theme.iconColor}`}>{key}</p>
                  <p className="text-zinc-400 leading-relaxed font-sans">{explanation}</p>
                </TooltipContent>
              </Tooltip>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="relative group/config mt-1 bg-[#09090b]/80 border border-white/5 rounded-lg p-4 flex flex-col gap-2 max-h-[350px] overflow-y-auto shadow-inner bg-gradient-to-br from-black/40 to-transparent">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest px-1">
          Active Engine Parameters
        </span>
        <Button
          variant="secondary"
          size="sm"
          className="h-6 px-2 text-[10px] uppercase font-bold tracking-widest bg-zinc-800/80 hover:bg-zinc-700/80 text-zinc-300 transition-colors shadow-sm"
          onClick={handleCopy}
        >
          {copied ? <Check className="w-3 h-3 mr-1.5 text-emerald-400" /> : <Copy className="w-3 h-3 mr-1.5 text-sky-400" />}
          {copied ? "Copied!" : "Copy Payload"}
        </Button>
      </div>

      <TooltipProvider>
        {renderGroup("Hardware & Distribution", hardware)}
        {renderGroup("Network Architecture", network)}
        {renderGroup("MCTS Search Logic", mcts)}
        {renderGroup("Training Hyperparameters", training)}
        {renderGroup("Other Setup", other)}
      </TooltipProvider>
    </div>
  );
}

// -----------------------------------------------------------------------------
// Compact Params Viewer for Optuna/Tune Dashboards
// -----------------------------------------------------------------------------
export function CompactTrialParams({ params }: { params: Record<string, any> }) {
  const [copied, setCopied] = useState(false);

  if (!params || Object.keys(params).length === 0) {
    return <span className="text-zinc-600 italic">No parameters</span>;
  }

  const { hardware, network, mcts, training, other } = categorizeParams(params);

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(JSON.stringify(params, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const renderCompactGroup = (title: string, items: [string, any][]) => {
    if (items.length === 0) return null;
    const theme = GROUP_THEMES[title] || GROUP_THEMES["Other Setup"];

    return (
      <>
        {items.map(([key, value]) => {
          const Icon = getIcon(key);
          return (
            <Tooltip key={key} delayDuration={100}>
              <TooltipTrigger asChild>
                <div className={`flex items-center gap-1.5 px-1.5 py-[2px] ${theme.badgeBg} border ${theme.badgeBorder} rounded text-[9px] font-mono cursor-help hover:brightness-125 transition-all truncate`}>
                  <Icon className={`w-3 h-3 ${theme.iconColor}`} />
                  <span className={`${theme.labelText} font-medium`}>{key}:</span>
                  <span className="font-bold text-white drop-shadow-md">{renderValue(value)}</span>
                </div>
              </TooltipTrigger>
              <TooltipContent side="top" className="bg-zinc-950 border-white/10 text-[10px] p-2 shadow-xl">
                <span className={theme.iconColor}>{title}</span>
              </TooltipContent>
            </Tooltip>
          );
        })}
      </>
    );
  };

  return (
    <TooltipProvider>
      <div className="flex items-start gap-2">
        <div className="flex flex-wrap gap-1 max-w-[400px]">
          {renderCompactGroup("Hardware & Distribution", hardware)}
          {renderCompactGroup("Network Architecture", network)}
          {renderCompactGroup("MCTS Search Logic", mcts)}
          {renderCompactGroup("Training Hyperparameters", training)}
          {renderCompactGroup("Other Setup", other)}
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={handleCopy}
          className="h-6 w-6 mt-0.5 shrink-0 hover:bg-white/10"
        >
          {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3 text-zinc-400" />}
        </Button>
      </div>
    </TooltipProvider>
  );
}

