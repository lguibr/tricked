import {
  VscSettingsGear,
  VscTypeHierarchy,
  VscServerProcess,
  VscRepoForked,
  VscLightbulb,
  VscPulse,
} from "react-icons/vsc";
import { GroupDef } from "@/components/execution/ParameterForm";

export const getSingleGroups = (_groupPresets: number[]): GroupDef[] => [
  {
    title: "Optimizer Global Controls",
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

];

export const getBoundGroups = (groupPresets: number[]): GroupDef[] => [
  {
    title: "1. Neural Topology & Width",
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
      {
        key: "value_support_size",
        label: "Value Support Size",
        min: 10,
        max: 600,
        step: 10,
        tooltip:
          "Size of the categorical support vector for value and reward prediction.",
      },
    ],
  },
  {
    title: "1B. MDP & Memory Depth",
    color: "text-amber-400",
    icon: VscSettingsGear,
    presetLevel: Math.max(1, groupPresets[0]),
    fields: [
      {
        key: "buffer_capacity_limit",
        label: "Replay Buffer Capacity",
        min: 1000,
        max: 1000000,
        step: 5000,
        tooltip: "Maximum number of game states to store in memory.",
      },
      {
        key: "unroll_steps",
        label: "Unroll Steps",
        min: 1,
        max: 20,
        step: 1,
        tooltip:
          "Number of steps unrolled in the recurrent dynamics network.",
      },
      {
        key: "temporal_difference_steps",
        label: "TD Steps",
        min: 1,
        max: 20,
        step: 1,
        tooltip: "n-step return horizon for training value targets.",
      },
    ],
  },
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
        tooltip: "Optimizer will search this discount factor space.",
      },
      {
        key: "td_lambda",
        label: "TD Lambda Range",
        min: 0.5,
        max: 1.0,
        step: 0.01,
        tooltip: "Optimizer will search this TD lambda space.",
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
        tooltip: "Optimizer will search this learning rate space.",
      },
      {
        key: "weight_decay",
        label: "Weight Decay Range",
        min: 0.0,
        max: 0.1,
        step: 0.0001,
        tooltip: "Optimizer will search this L2 regularization space.",
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
