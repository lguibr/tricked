import { useEffect, useState, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { MetricChart } from "./dashboard/MetricChart";

import { MctsTreeGraph } from "./execution/MctsTreeGraph";
import { HexagonalHeatmap } from "./execution/HexagonalHeatmap";
import { LossStackedArea } from "./execution/LossStackedArea";
import { ActionThemeRiver } from "./execution/ActionThemeRiver";
import { TdErrorWaterfall } from "./execution/TdErrorWaterfall";
import { ReplayBufferBar } from "./execution/ReplayBufferBar";

import type { Run } from "@/bindings/Run";

interface MetricsDashboardProps {
  runs: Run[];
  runIds: string[];
  runColors: Record<string, string>;
}

export function MetricsDashboard({
  runs,
  runIds,
  runColors,
}: MetricsDashboardProps) {
  const metricsDataRef = useRef<Record<string, any[]>>({});
  const [xAxisMode, setXAxisMode] = useState<"step" | "relative" | "absolute">(
    "step",
  );

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
        // Merge fetched base metrics
        metricsDataRef.current = { ...metricsDataRef.current, ...data };
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000); // Polling as fallback, UDP takes priority

    // Subscribe to live UDP metrics
    import("@tauri-apps/api/event").then(({ listen }) => {
      listen("engine_telemetry", (event: any) => {
        if (!active) return;
        const metric = event.payload;
        if (runIds.includes(metric.run_id)) {
          const currentArr = metricsDataRef.current[metric.run_id] || [];
          // Avoid duplicate steps
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

  const charts = [
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
      key: "lr",
      title: "LEARNING RATE",
      description: "Step size for the optimizer. Decays via cosine annealing.",
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
  ];

  return (
    <div className="flex flex-col h-full w-full bg-border/20">
      {/* Header X-Axis Toggle */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-border/20 bg-background/50 shrink-0">
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

      {/* Grid and Deep Observability Container */}
      <div className="flex-1 flex flex-col bg-border/50 overflow-y-auto">
        <div className="grid grid-cols-4 gap-[1px] auto-rows-[250px] shrink-0 mb-[1px]">
          {charts.map((chart) => (
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

        {/* Deep Observability Section */}
        <div className="bg-background w-full py-3 px-6 border-y border-border/20 mt-4 mb-[1px] shrink-0">
          <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-widest">
            Deep Observability
          </h3>
          <p className="text-[10px] text-zinc-600 mt-1">
            High-dimensional and high-frequency visualizations for advanced
            reinforcement learning analysis.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-[1px] auto-rows-[300px] shrink-0 pb-12">
          <div className="bg-background p-1">
            <MctsTreeGraph
              runs={runs}
              runIds={runIds}
              metricsDataRef={metricsDataRef}
              runColors={runColors}
            />
          </div>
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
            <TdErrorWaterfall
              runs={runs}
              runIds={runIds}
              metricsDataRef={metricsDataRef}
              runColors={runColors}
            />
          </div>
          <div className="bg-background p-1">
            <ReplayBufferBar
              runs={runs}
              runIds={runIds}
              metricsDataRef={metricsDataRef}
              runColors={runColors}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
