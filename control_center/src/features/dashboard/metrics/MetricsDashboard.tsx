import { useState } from "react";
import {
  VscChevronDown,
  VscChevronRight,
  VscGraph,
  VscTypeHierarchy,
  VscServerProcess,
  VscEye,
} from "react-icons/vsc";
import { MetricChart } from "@/components/dashboard/MetricChart";
import { HexagonalHeatmap } from "@/components/execution/HexagonalHeatmap";
import { LossStackedArea } from "@/components/execution/LossStackedArea";
import { LayerNormsDisplay } from "@/components/dashboard/LayerNormsDisplay";
import {
  neuralCharts,
  agentCharts,
  systemCharts,
} from "@/components/dashboard/MetricsMappings";
import { useAppStore } from "@/store/useAppStore";
import { useMetricsData } from "./useMetricsData";
import { MetricsHeader } from "./MetricsHeader";

export function MetricsDashboard({
  inWorkspace = false,
}: {
  inWorkspace?: boolean;
}) {
  const runs = useAppStore((state: any) => state.runs);
  const runIds = useAppStore((state: any) => state.selectedDashboardRuns);
  const runColors = useAppStore((state: any) => state.runColors);
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

  const metricsDataRef = useMetricsData(runIds);

  const toggleSection = (section: keyof typeof expanded) => {
    setExpanded((prev) => ({ ...prev, [section]: !prev[section] }));
  };

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
      <MetricsHeader
        smoothingWeight={smoothingWeight}
        setSmoothingWeight={setSmoothingWeight}
        xAxisMode={xAxisMode}
        setXAxisMode={setXAxisMode}
      />

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
              <div className="grid grid-cols-3 gap-[1px] auto-rows-[220px] shrink-0 bg-white/5">
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
              <div className="grid grid-cols-3 gap-[1px] auto-rows-[220px] shrink-0 bg-white/5">
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
              <div className="grid grid-cols-3 gap-[1px] auto-rows-[220px] shrink-0 bg-white/5">
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
              <div className="flex flex-col gap-[1px] shrink-0 bg-white/5 pb-2">
                <div className="bg-[#050505] w-full min-h-[450px]">
                  <LossStackedArea
                    runs={runs}
                    runIds={runIds}
                    metricsDataRef={metricsDataRef}
                    runColors={runColors}
                    xAxisMode={xAxisMode}
                    smoothingWeight={smoothingWeight}
                  />
                </div>
                <div className="bg-[#050505] w-full min-h-[500px] h-[70vh] max-h-[70vh]">
                  <HexagonalHeatmap
                    runs={runs}
                    runIds={runIds}
                    metricsDataRef={metricsDataRef}
                    runColors={runColors}
                  />
                </div>
                <div className="bg-[#050505] w-full min-h-[300px]">
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
