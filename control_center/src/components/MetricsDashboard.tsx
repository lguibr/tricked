import { useEffect, useState, useRef } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import { MetricChart } from "./dashboard/MetricChart";

import { HexagonalHeatmap } from "./execution/HexagonalHeatmap";
import { LossStackedArea } from "./execution/LossStackedArea";
import { ActionThemeRiver } from "./execution/ActionThemeRiver";
import * as echarts from "echarts";

import type { Run } from "@/bindings/Run";

const LayerNormsDisplay = ({ runIds, metricsDataRef }: any) => {
  const [data, setData] = useState<string>("Waiting for telemetry...");

  useEffect(() => {
    let unmounted = false;
    const interval = setInterval(() => {
      if (unmounted) return;
      const latest = runIds
        .map((id: string) => {
          const arr = metricsDataRef.current[id];
          return arr && arr.length > 0
            ? arr[arr.length - 1].layer_gradient_norms
            : null;
        })
        .filter(Boolean);

      if (latest.length > 0) {
        setData(latest[latest.length - 1]);
      }
    }, 1000);
    return () => {
      unmounted = true;
      clearInterval(interval);
    };
  }, [runIds, metricsDataRef]);

  return (
    <div className="flex flex-col h-full w-full bg-background border border-border/20 rounded-md p-3">
      <div className="text-[10px] font-bold text-emerald-400 uppercase tracking-widest mb-2 flex items-center gap-1">
        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
        Layer-Wise Gradient Norms (Top 3)
      </div>
      <div className="flex-1 flex items-center justify-center font-mono text-[10px] text-zinc-400 text-center px-4 overflow-y-auto">
        {data || "N/A"}
      </div>
    </div>
  );
};

interface MetricsDashboardProps {
  runs: Run[];
  runIds: string[];
  runColors: Record<string, string>;
}

// inside the component:
export function MetricsDashboard({
  runs,
  runIds,
  runColors,
}: MetricsDashboardProps) {
  const metricsDataRef = useRef<Record<string, any[]>>({});
  const [xAxisMode, setXAxisMode] = useState<"step" | "relative" | "absolute">(
    "step",
  );

  const [expanded, setExpanded] = useState({
    neural: true,
    agent: true,
    system: true,
    deep: true,
  });

  const toggleSection = (section: keyof typeof expanded) => {
    setExpanded((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  useEffect(() => {
    echarts.connect("metricsGroup");
    return () => echarts.disconnect("metricsGroup");
  }, []);

  useEffect(() => {
    let active = true;
    let unlisten: (() => void) | undefined;

    const fetchMetrics = async () => {
      const data: Record<string, any[]> = {};
      for (const id of runIds) {
        try {
          const runMetrics = await invoke<any[]>("get_run_metrics", { id });
          data[id] = runMetrics;
        } catch (e) {
          console.error(`Failed to fetch metrics for ${id}:`, e);
        }
      }
      if (active) {
        metricsDataRef.current = { ...metricsDataRef.current, ...data };
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000);

    import("@tauri-apps/api/event").then(({ listen }) => {
      listen("engine_telemetry", (event: any) => {
        if (!active) return;
        const metric = event.payload;
        if (runIds.includes(metric.run_id)) {
          const currentArr = metricsDataRef.current[metric.run_id] || [];
          if (!currentArr.some((m) => m.step === metric.step)) {
            metricsDataRef.current[metric.run_id] = [...currentArr, metric];
          }
        }
      }).then((u) => {
        unlisten = u;
      });
    });

    return () => {
      active = false;
      clearInterval(interval);
      if (unlisten) unlisten();
    };
  }, [runIds]);

  const allCharts = [
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
      description:
        "Error evaluating the game state advantages. Lower is better.",
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
      description:
        "Average game score attained in self-play. Higher is better.",
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
      title: "VRAM USAGE (MB)",
      description: "GPU memory consumption.",
    },
    {
      key: "ram_usage_mb",
      title: "RAM USAGE (MB)",
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
      title: "QUEUE LATENCY (μs)",
      description:
        "Average time requests spend waiting in the inference queue.",
    },
    {
      key: "sumtree_contention_us",
      title: "SUMTREE CONTENTION (μs)",
      description: "Time spent blocking on SumTree shard locks during updates.",
    },
    {
      key: "sps_vs_tps",
      title: "SPS VS TPS",
      description: "Ratio of transitions trained to simulations generated.",
    },
  ];

  const neuralCharts = allCharts.filter((c) =>
    [
      "total_loss",
      "policy_loss",
      "value_loss",
      "value_prefix_loss",
      "policy_entropy",
      "action_space_entropy",
      "gradient_norm",
      "representation_drift",
      "mean_td_error",
    ].includes(c.key),
  );
  const agentCharts = allCharts.filter((c) =>
    [
      "game_score_mean",
      "game_lines_cleared",
      "mcts_depth_mean",
      "mcts_search_time_mean",
    ].includes(c.key),
  );
  const systemCharts = allCharts.filter((c) =>
    [
      "lr",
      "gpu_usage_pct",
      "cpu_usage_pct",
      "vram_usage_mb",
      "ram_usage_mb",
      "disk_usage_pct",
      "queue_saturation_ratio",
      "queue_latency_us",
      "sumtree_contention_us",
      "sps_vs_tps",
    ].includes(c.key),
  );

  const renderSectionHeader = (
    title: string,
    sectionKey: keyof typeof expanded,
    color: string,
  ) => (
    <div
      className="bg-black/40 hover:bg-black/60 cursor-pointer w-full py-3 px-6 border-b border-border/20 flex items-center justify-between shrink-0 transition-colors"
      onClick={() => toggleSection(sectionKey)}
    >
      <div className="flex items-center gap-2">
        {expanded[sectionKey] ? (
          <ChevronDown className="w-4 h-4 text-zinc-500" />
        ) : (
          <ChevronRight className="w-4 h-4 text-zinc-500" />
        )}
        <h3 className={`text-xs font-bold ${color} uppercase tracking-widest`}>
          {title}
        </h3>
      </div>
    </div>
  );

  return (
    <div className="flex flex-col h-full w-full bg-border/10 overflow-y-auto">
      {/* Header X-Axis Toggle */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-border/20 bg-background/95 sticky top-0 z-10 shrink-0">
        <h2 className="text-xs font-bold text-zinc-100 uppercase tracking-widest">
          Engine Observability
        </h2>
        <div className="flex items-center space-x-2">
          <span className="text-[10px] font-bold uppercase tracking-widest text-zinc-500 mr-2">
            X-Axis Mode
          </span>
          <div className="flex bg-black/50 p-1 rounded-lg border border-border/10">
            <button
              onClick={() => setXAxisMode("step")}
              className={`px-3 py-1 text-[11px] uppercase tracking-wider font-semibold rounded-md transition-colors ${xAxisMode === "step" ? "bg-primary/20 text-primary border border-primary/50 shadow-sm" : "text-zinc-500 hover:text-zinc-300 border border-transparent"}`}
            >
              Step
            </button>
            <button
              onClick={() => setXAxisMode("relative")}
              className={`px-3 py-1 text-[11px] uppercase tracking-wider font-semibold rounded-md transition-colors ${xAxisMode === "relative" ? "bg-primary/20 text-primary border border-primary/50 shadow-sm" : "text-zinc-500 hover:text-zinc-300 border border-transparent"}`}
            >
              Relative Time
            </button>
            <button
              onClick={() => setXAxisMode("absolute")}
              className={`px-3 py-1 text-[11px] uppercase tracking-wider font-semibold rounded-md transition-colors ${xAxisMode === "absolute" ? "bg-primary/20 text-primary border border-primary/50 shadow-sm" : "text-zinc-500 hover:text-zinc-300 border border-transparent"}`}
            >
              Absolute Time
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 flex flex-col pb-12">
        {/* A. Neural & Gradient Dynamics */}
        {renderSectionHeader(
          "A. Neural & Gradient Dynamics",
          "neural",
          "text-purple-400",
        )}
        {expanded.neural && (
          <div className="grid grid-cols-4 gap-[1px] auto-rows-[250px] shrink-0 bg-border/20">
            {neuralCharts.map((chart) => (
              <div key={chart.key} className="bg-background">
                <MetricChart
                  title={chart.title}
                  description={chart.description}
                  metricKey={chart.key}
                  runs={runs}
                  runIds={runIds}
                  metricsDataRef={metricsDataRef}
                  runColors={runColors}
                  xAxisMode={xAxisMode}
                />
              </div>
            ))}
          </div>
        )}

        {/* B. Agent Performance & MDP */}
        {renderSectionHeader(
          "B. Agent Performance & MDP",
          "agent",
          "text-blue-400",
        )}
        {expanded.agent && (
          <div className="grid grid-cols-4 gap-[1px] auto-rows-[250px] shrink-0 bg-border/20">
            {agentCharts.map((chart) => (
              <div key={chart.key} className="bg-background">
                <MetricChart
                  title={chart.title}
                  description={chart.description}
                  metricKey={chart.key}
                  runs={runs}
                  runIds={runIds}
                  metricsDataRef={metricsDataRef}
                  runColors={runColors}
                  xAxisMode={xAxisMode}
                />
              </div>
            ))}
          </div>
        )}

        {/* C. Systems & Hardware Utilization */}
        {renderSectionHeader(
          "C. Systems & Hardware Utilization",
          "system",
          "text-amber-400",
        )}
        {expanded.system && (
          <div className="grid grid-cols-4 gap-[1px] auto-rows-[250px] shrink-0 bg-border/20">
            {systemCharts.map((chart) => (
              <div key={chart.key} className="bg-background">
                <MetricChart
                  title={chart.title}
                  description={chart.description}
                  metricKey={chart.key}
                  runs={runs}
                  runIds={runIds}
                  metricsDataRef={metricsDataRef}
                  runColors={runColors}
                  xAxisMode={xAxisMode}
                />
              </div>
            ))}
          </div>
        )}

        {/* D. Deep Observability & Heatmaps */}
        {renderSectionHeader(
          "D. Deep Observability & Heatmaps",
          "deep",
          "text-emerald-400",
        )}
        {expanded.deep && (
          <div className="grid grid-cols-2 gap-[1px] auto-rows-[300px] shrink-0 bg-border/20">
            <div className="bg-background p-1">
              <HexagonalHeatmap
                runs={runs}
                runIds={runIds}
                metricsDataRef={metricsDataRef}
                runColors={runColors}
              />
            </div>
            <div className="bg-background p-1">
              <LossStackedArea
                runs={runs}
                runIds={runIds}
                metricsDataRef={metricsDataRef}
                runColors={runColors}
              />
            </div>
            <div className="bg-background p-1">
              <ActionThemeRiver
                runs={runs}
                runIds={runIds}
                metricsDataRef={metricsDataRef}
                runColors={runColors}
              />
            </div>
            <div className="bg-background p-1">
              <LayerNormsDisplay
                runs={runs}
                runIds={runIds}
                metricsDataRef={metricsDataRef}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
