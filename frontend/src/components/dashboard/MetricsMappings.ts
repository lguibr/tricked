export const allCharts = [
  {
    key: "total_loss",
    title: "TOTAL LOSS",
    description: "Overall loss. Should trend downwards over time.",
  },
  {
    key: "policy_loss",
    title: "POLICY LOSS",
    description:
      "Error predicting the best move probabilities. Lower is better.",
  },
  {
    key: "value_loss",
    title: "VALUE LOSS",
    description: "Error evaluating the game state advantages. Lower is better.",
  },
  {
    key: "value_prefix_loss",
    title: "VALUE PREFIX LOSS",
    description: "Error predicting sequential game rewards. Lower is better.",
  },
  {
    key: "policy_entropy",
    title: "POLICY ENTROPY",
    description: "Randomness of the policy network's output distribution.",
  },
  {
    key: "gradient_norm",
    title: "GRADIENT NORM",
    description: "Magnitude of the gradients before clipping.",
  },
  {
    key: "action_space_entropy",
    title: "ACTION ENTROPY",
    description:
      "Entropy of the MCTS action distribution. Higher means more exploration.",
  },
  {
    key: "representation_drift",
    title: "REPRESENTATION DRIFT",
    description:
      "Cosine similarity divergence between active and EMA representations.",
  },
  {
    key: "mean_td_error",
    title: "MEAN TD ERROR",
    description:
      "Mean Temporal Difference error indicating value prediction accuracy.",
  },
  {
    key: "game_score_mean",
    title: "SCORE MEAN",
    description: "Average game score attained in self-play. Higher is better.",
  },
  {
    key: "game_lines_cleared",
    title: "LINES CLEARED",
    description: "Average clear count in the environment. Higher is better.",
  },
  {
    key: "mcts_depth_mean",
    title: "MCTS DEPTH MEAN",
    description: "Mean depth of the search tree. Indicates tactical horizon.",
  },
  {
    key: "mcts_search_time_mean",
    title: "MCTS TIME (ms)",
    description: "Search iteration time. Indicates GPU/CPU bottlenecks.",
  },
  {
    key: "lr",
    title: "LEARNING RATE",
    description: "Step size for the optimizer. Decays via cosine annealing.",
  },
  {
    key: "gpu_usage_pct",
    title: "GPU USAGE %",
    description: "GPU compute saturation. Goal is >95%.",
  },
  {
    key: "cpu_usage_pct",
    title: "CPU USAGE %",
    description: "Total CPU core saturation across workers.",
  },
  {
    key: "vram_usage_mb",
    title: "VRAM (MB)",
    description: "GPU memory consumption.",
  },
  {
    key: "ram_usage_mb",
    title: "RAM (MB)",
    description: "System memory consumption.",
  },
  {
    key: "disk_usage_pct",
    title: "DISK USAGE %",
    description: "Current disk saturation for checkpoints.",
  },
  {
    key: "queue_saturation_ratio",
    title: "QUEUE SATURATION",
    description: "Ratio of inference batch fullness vs maximum limit.",
  },
  {
    key: "queue_latency_us",
    title: "LATENCY (μs)",
    description: "Average time requests spend waiting in the inference queue.",
  },
  {
    key: "sumtree_contention_us",
    title: "CONTENTION (μs)",
    description: "Time spent blocking on SumTree shard locks during updates.",
  },
  {
    key: "sps_vs_tps",
    title: "SPS VS TPS",
    description: "Ratio of transitions trained to simulations generated.",
  },
  {
    key: "network_tx_mbps",
    title: "NETWORK TX (Mbps)",
    description: "Network transmit bandwidth saturation.",
  },
  {
    key: "network_rx_mbps",
    title: "NETWORK RX (Mbps)",
    description: "Network receive bandwidth saturation.",
  },
  {
    key: "disk_read_mbps",
    title: "DISK READ (MB/s)",
    description: "Disk read throughput across workers.",
  },
  {
    key: "disk_write_mbps",
    title: "DISK WRITE (MB/s)",
    description: "Disk write throughput for artifacts.",
  },
  {
    key: "difficulty",
    title: "CURRICULUM LEVEL",
    description: "Current shape complexity topology the agent is playing.",
  },
];

export const neuralCharts = allCharts.filter((c) =>
  [
    "total_loss",
    "policy_loss",
    "value_loss",
    "policy_entropy",
    "action_space_entropy",
    "gradient_norm",
    "representation_drift",
    "mean_td_error",
  ].includes(c.key),
);

export const agentCharts = allCharts.filter((c) =>
  [
    "difficulty",
    "game_score_mean",
    "game_lines_cleared",
    "mcts_depth_mean",
    "mcts_search_time_mean",
  ].includes(c.key),
);

export const systemCharts = allCharts.filter((c) =>
  [
    "lr",
    "gpu_usage_pct",
    "cpu_usage_pct",
    "vram_usage_mb",
    "ram_usage_mb",
    "disk_usage_pct",
    "network_tx_mbps",
    "network_rx_mbps",
    "disk_read_mbps",
    "disk_write_mbps",
    "queue_saturation_ratio",
    "queue_latency_us",
    "sumtree_contention_us",
  ].includes(c.key),
);
