import {
  Cpu,
  Layers,
  Brain,
  Clock,
  Activity,
  TrendingUp,
  Repeat,
  Zap,
  HardDrive,
  Users,
} from "lucide-react";
import React from "react";

export const EXPLANATIONS: Record<string, string> = {
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

export const getIcon = (key: string) => {
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

export const categorizeParams = (configObj: Record<string, any>) => {
  const HARDWARE = ["device", "worker_device", "num_processes", "inference_batch_size_limit", "checkpoint_interval", "checkpoint_history"];
  const NETWORK = ["num_blocks", "hidden_dimension_size", "support_size", "resnetBlocks", "resnetChannels", "value_support_size", "reward_support_size", "spatial_channel_count", "hole_predictor_dim"];
  const MCTS = ["simulations", "max_gumbel_k", "inference_timeout_ms"];
  const TRAINING = ["lr_init", "train_batch_size", "buffer_capacity_limit", "unroll_steps", "temporal_difference_steps", "discount_factor", "td_lambda", "weight_decay", "reanalyze_ratio"];

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

export const GROUP_THEMES: Record<string, { theme: string; iconColor: string; badgeBg: string; badgeBorder: string; labelText: string }> = {
  "Hardware & Distribution": {
    theme: "amber",
    iconColor: "text-amber-400",
    badgeBg: "bg-amber-500/10",
    badgeBorder: "border-amber-500/20",
    labelText: "text-amber-500/80",
  },
  "Network Architecture": {
    theme: "purple",
    iconColor: "text-purple-400",
    badgeBg: "bg-purple-500/10",
    badgeBorder: "border-purple-500/20",
    labelText: "text-purple-500/80",
  },
  "MCTS Search Logic": {
    theme: "blue",
    iconColor: "text-blue-400",
    badgeBg: "bg-blue-500/10",
    badgeBorder: "border-blue-500/20",
    labelText: "text-blue-500/80",
  },
  "Training Hyperparameters": {
    theme: "emerald",
    iconColor: "text-emerald-400",
    badgeBg: "bg-emerald-500/10",
    badgeBorder: "border-emerald-500/20",
    labelText: "text-emerald-500/80",
  },
  "Other Setup": {
    theme: "zinc",
    iconColor: "text-zinc-400",
    badgeBg: "bg-zinc-500/10",
    badgeBorder: "border-zinc-500/20",
    labelText: "text-zinc-500/80",
  },
};

export const renderValue = (val: any) => {
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
