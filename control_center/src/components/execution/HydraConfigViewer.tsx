import { Copy } from "lucide-react";
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
  train_epochs:
    "Number of epochs to train the network on the gathered experience.",
  worker_device: "Compute device assigned to the rollout workers.",
  unroll_steps:
    "Number of steps into the future the network is trained to predict.",
  temporal_difference_steps: "TD-steps (n-step return) used for value targets.",
  inference_batch_size_limit:
    "Max batch size for parallel inference requests from workers.",
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
        <span className={val ? "text-green-400" : "text-red-400"}>
          {val ? "true" : "false"}
        </span>
      );
    }
    if (typeof val === "number") {
      return <span className="text-blue-400">{val}</span>;
    }
    if (typeof val === "string") {
      return <span className="text-yellow-300">"{val}"</span>;
    }
    if (Array.isArray(val)) {
      return <span className="text-purple-400">[{val.join(", ")}]</span>;
    }
    return <span className="text-zinc-500">{JSON.stringify(val)}</span>;
  };

  return (
    <div className="relative group/config mt-1 bg-[#0a0a0c] border border-border/20 rounded-md p-2 flex flex-col gap-1 max-h-[300px] overflow-y-auto">
      <TooltipProvider>
        <div className="grid grid-cols-1 gap-x-2 gap-y-1">
          {Object.entries(configObj).map(([key, value]) => (
            <div key={key} className="flex items-start text-[10px] sm:text-xs">
              <Tooltip delayDuration={300}>
                <TooltipTrigger asChild>
                  <span className="font-medium text-emerald-400/80 hover:text-emerald-400 cursor-help underline underline-offset-2 decoration-border/50 hover:decoration-emerald-500/50 break-all w-[45%] shrink-0 pr-2">
                    {key}
                  </span>
                </TooltipTrigger>
                <TooltipContent
                  side="right"
                  className="max-w-[200px] bg-zinc-900 border-border/50 text-xs text-zinc-200"
                >
                  <p>
                    {EXPLANATIONS[key] || `Configuration parameter: ${key}`}
                  </p>
                </TooltipContent>
              </Tooltip>
              <span className="text-zinc-600 mr-2">:</span>
              <span className="font-mono break-all">{renderValue(value)}</span>
            </div>
          ))}
        </div>
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
