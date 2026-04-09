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

import { LayerNormsDisplay } from "./dashboard/LayerNormsDisplay";
import {
  neuralCharts,
  agentCharts,
  systemCharts,
} from "./dashboard/MetricsMappings";

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
            runId: id,
          });
          const existing = metricsDataRef.current[id] || [];
          const merged = new Map<number, any>();
          for (const m of existing) merged.set(m.step, m);
          for (const m of runMetrics) merged.set(m.step, m);

          const finalArray = Array.from(merged.values()).sort(
            (a, b) => a.step - b.step,
          );
          data[id] = finalArray;
          console.warn(
            `[DEBUG] fetchMetrics for ${id}: runMetrics items = ${runMetrics?.length}, final merged size = ${finalArray.length}`,
          );
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
        {runIds.length === 0 ? (
          <div className="flex-1 flex items-center justify-center w-full h-full min-h-[300px]">
            <div className="flex flex-col items-center gap-3 p-6 bg-[#0a0a0c] border border-white/5 rounded-lg shadow-2xl opacity-60">
              <VscGraph className="w-8 h-8 text-zinc-600" />
              <span className="text-zinc-400 font-mono text-[10px] uppercase tracking-widest text-center">
                No Telemetry Sources Selected
                <br />
                <span className="text-[9px] text-zinc-600">
                  Select a run from the sidebar to replay metrics
                </span>
              </span>
            </div>
          </div>
        ) : (
          <>
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
          </>
        )}
      </div>
    </div>
  );
}
