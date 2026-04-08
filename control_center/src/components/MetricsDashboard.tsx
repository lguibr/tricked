import { useEffect, useState, useRef } from "react";
import {
  VscChevronDown,
  VscChevronRight,
  VscGraph,
  VscTypeHierarchy,
  VscServerProcess,
  VscEye,
} from "react-icons/vsc";
import { invoke } from "@tauri-apps/api/core";
import { MetricChart } from "./dashboard/MetricChart";

import { HexagonalHeatmap } from "./execution/HexagonalHeatmap";
import { LossStackedArea } from "./execution/LossStackedArea";
import { Slider } from "./ui/slider";
import * as echarts from "echarts";

const LayerNormsDisplay = ({ runIds, metricsDataRef }: any) => {
  const [data, setData] = useState<string>("Waiting for telemetry...");

  useEffect(() => {
    let unmounted = false;
    const interval = setInterval(() => {
      if (unmounted) return;
      const latest = runIds
        .map((id: string) => {
          const arr = metricsDataRef.current[id];
          if (!arr || arr.length === 0) return null;
          for (let i = arr.length - 1; i >= 0; i--) {
            if (arr[i].layer_gradient_norms) {
              return arr[i].layer_gradient_norms;
            }
          }
          return null;
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
    <div className="flex flex-col h-full w-full bg-[#050505] p-2">
      <div className="text-[9.5px] font-bold text-emerald-400 uppercase tracking-widest mb-1 flex items-center gap-1.5 border-b border-white/5 pb-1">
        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_5px_rgba(16,185,129,0.5)]" />
        Layer-Wise Gradient Norms (Top 3)
      </div>
      <div className="flex-1 flex items-center justify-center font-mono text-[9px] text-zinc-400 text-center px-2 overflow-y-auto leading-tight">
        {data || "N/A"}
      </div>
    </div>
  );
};

import { useAppStore } from "@/store/useAppStore";

export function MetricsDashboard({
  inWorkspace = false,
}: {
  inWorkspace?: boolean;
}) {
  const runs = useAppStore((state: any) => state.runs);
  const runIds = useAppStore((state: any) => state.selectedDashboardRuns);
  const runColors = useAppStore((state: any) => state.runColors);
  const metricsDataRef = useRef<Record<string, any[]>>({});
  const [xAxisMode, setXAxisMode] = useState<"step" | "relative" | "absolute">(
    "step",
  );
  const [smoothingWeight, setSmoothingWeight] = useState<number>(0.9);

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
          const runMetrics = await invoke<any[]>("get_run_metrics", {
            run_id: id,
          });
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
      listen("live_metric", (event: any) => {
        if (!active) return;
        const metric = event.payload;
        if (runIds.includes(metric.run_id)) {
          const currentArr = metricsDataRef.current[metric.run_id] || [];
          if (!currentArr.some((m) => m.step === metric.step)) {
            metricsDataRef.current[metric.run_id] = [...currentArr, metric];
          }
        }
      }).then((u: any) => {
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
      description:
        "Average time requests spend waiting in the inference queue.",
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
      key: "difficulty",
      title: "CURRICULUM LEVEL",
      description: "Current shape complexity topology the agent is playing.",
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
      "difficulty",
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
    colorClass: string,
    Icon: any,
  ) => (
    <div
      className="bg-[#0a0a0c] hover:bg-[#111113] cursor-pointer w-full py-1.5 px-3 border-y border-white/5 flex items-center justify-between shrink-0 transition-colors shadow-sm"
      onClick={() => toggleSection(sectionKey)}
    >
      <div className="flex items-center gap-1.5">
        {expanded[sectionKey] ? (
          <VscChevronDown className="w-3.5 h-3.5 text-zinc-500" />
        ) : (
          <VscChevronRight className="w-3.5 h-3.5 text-zinc-500" />
        )}
        <Icon className={`w-3.5 h-3.5 ${colorClass}`} />
        <h3
          className={`text-[9.5px] font-black ${colorClass} uppercase tracking-widest`}
        >
          {title}
        </h3>
      </div>
    </div>
  );

  return (
    <div
      className={`flex flex-col h-full w-full overflow-y-auto custom-scrollbar relative ${inWorkspace ? "bg-transparent" : "bg-[#020202]"}`}
    >
      {/* Header X-Axis Toggle */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/10 bg-[#080808]/95 backdrop-blur-md sticky top-0 z-20 shrink-0">
        <h2 className="text-[10px] font-bold text-zinc-200 uppercase tracking-widest flex items-center gap-1.5">
          <VscGraph className="w-3.5 h-3.5 text-primary" />
          Engine Observability
        </h2>
        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-2">
            <span className="text-[8.5px] font-bold uppercase tracking-widest text-zinc-500">
              SMOOTHING
            </span>
            <div className="w-24 px-1 flex items-center">
              <Slider
                defaultValue={[smoothingWeight]}
                max={0.999}
                min={0.0}
                step={0.001}
                onValueChange={(val: any) => setSmoothingWeight(val[0])}
              />
            </div>
            <span className="text-[8px] font-mono text-zinc-400 w-6 text-right">
              {smoothingWeight.toFixed(2)}
            </span>
          </div>

          <div className="flex items-center space-x-2">
            <span className="text-[8.5px] font-bold uppercase tracking-widest text-zinc-500">
              X-Axis
            </span>
            <div className="flex bg-black/80 p-0.5 rounded border border-white/10 shadow-inner">
              <button
                onClick={() => setXAxisMode("step")}
                className={`px-2 py-0.5 text-[8.5px] uppercase tracking-widest font-black rounded-sm transition-all ${xAxisMode === "step" ? "bg-primary/20 text-primary border border-primary/40 shadow-sm" : "text-zinc-500 hover:text-zinc-300 border border-transparent hover:bg-white/5"}`}
              >
                Step
              </button>
              <button
                onClick={() => setXAxisMode("relative")}
                className={`px-2 py-0.5 text-[8.5px] uppercase tracking-widest font-black rounded-sm transition-all ${xAxisMode === "relative" ? "bg-primary/20 text-primary border border-primary/40 shadow-sm" : "text-zinc-500 hover:text-zinc-300 border border-transparent hover:bg-white/5"}`}
              >
                Relative
              </button>
              <button
                onClick={() => setXAxisMode("absolute")}
                className={`px-2 py-0.5 text-[8.5px] uppercase tracking-widest font-black rounded-sm transition-all ${xAxisMode === "absolute" ? "bg-primary/20 text-primary border border-primary/40 shadow-sm" : "text-zinc-500 hover:text-zinc-300 border border-transparent hover:bg-white/5"}`}
              >
                Absolute
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 flex flex-col pb-8">
        {renderSectionHeader(
          "A. Neural & Gradient Dynamics",
          "neural",
          "text-purple-400",
          VscTypeHierarchy,
        )}
        {expanded.neural && (
          <div className="grid grid-cols-4 gap-[1px] auto-rows-[220px] shrink-0 bg-white/5">
            {neuralCharts.map((chart, index) => (
              <div key={chart.key} className="bg-[#050505] w-full h-full">
                <MetricChart
                  title={chart.title}
                  description={chart.description}
                  metricKey={chart.key}
                  runs={runs}
                  runIds={runIds}
                  metricsDataRef={metricsDataRef}
                  runColors={runColors}
                  xAxisMode={xAxisMode}
                  metricIndex={index}
                  smoothingWeight={smoothingWeight}
                />
              </div>
            ))}
          </div>
        )}

        {renderSectionHeader(
          "B. Agent Performance & MDP",
          "agent",
          "text-blue-400",
          VscGraph,
        )}
        {expanded.agent && (
          <div className="grid grid-cols-4 gap-[1px] auto-rows-[220px] shrink-0 bg-white/5">
            {agentCharts.map((chart, index) => (
              <div key={chart.key} className="bg-[#050505] w-full h-full">
                <MetricChart
                  title={chart.title}
                  description={chart.description}
                  metricKey={chart.key}
                  runs={runs}
                  runIds={runIds}
                  metricsDataRef={metricsDataRef}
                  runColors={runColors}
                  xAxisMode={xAxisMode}
                  metricIndex={index}
                  smoothingWeight={smoothingWeight}
                />
              </div>
            ))}
          </div>
        )}

        {renderSectionHeader(
          "C. Systems & Hardware Utilization",
          "system",
          "text-amber-400",
          VscServerProcess,
        )}
        {expanded.system && (
          <div className="grid grid-cols-4 gap-[1px] auto-rows-[220px] shrink-0 bg-white/5">
            {systemCharts.map((chart, index) => (
              <div key={chart.key} className="bg-[#050505] w-full h-full">
                <MetricChart
                  title={chart.title}
                  description={chart.description}
                  metricKey={chart.key}
                  runs={runs}
                  runIds={runIds}
                  metricsDataRef={metricsDataRef}
                  runColors={runColors}
                  xAxisMode={xAxisMode}
                  metricIndex={index}
                  smoothingWeight={smoothingWeight}
                />
              </div>
            ))}
          </div>
        )}

        {renderSectionHeader(
          "D. Deep Observability & Heatmaps",
          "deep",
          "text-emerald-400",
          VscEye,
        )}
        {expanded.deep && (
          <div className="grid grid-cols-2 gap-[1px] auto-rows-[270px] shrink-0 bg-white/5">
            <div className="bg-[#050505] w-full h-full">
              <HexagonalHeatmap
                runs={runs}
                runIds={runIds}
                metricsDataRef={metricsDataRef}
                runColors={runColors}
              />
            </div>
            <div className="bg-[#050505] w-full h-full">
              <LossStackedArea
                runs={runs}
                runIds={runIds}
                metricsDataRef={metricsDataRef}
                runColors={runColors}
              />
            </div>
            <div className="bg-[#050505] w-full h-full">
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
