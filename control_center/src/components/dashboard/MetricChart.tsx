import { useCallback } from "react";
import ReactECharts from "echarts-for-react";
import { Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { Run } from "@/bindings/Run";
import { useChartSync } from "@/hooks/useChartSync";
import { computePinnedSnapshots, getSharedXAxisConfig } from "@/lib/chart-utils";

interface MetricChartProps {
  title: string;
  description: string;
  metricKey: string;
  runs: Run[];
  runIds: string[];
  metricsDataRef: React.MutableRefObject<Record<string, any[]>>;
  runColors: Record<string, string>;
  xAxisMode?: "step" | "relative" | "absolute";

  smoothingWeight?: number;
}

export function MetricChart({
  title,
  description,
  metricKey,
  runs,
  runIds,
  metricsDataRef,
  runColors,
  xAxisMode = "step",

  smoothingWeight = 0.9,
}: MetricChartProps) {
  
  const getXAxisConfigCb = useCallback((pinnedTime: number | null) => {
    const snaps = pinnedTime !== null 
      ? computePinnedSnapshots(runIds, runs, metricsDataRef, xAxisMode, pinnedTime, [metricKey], smoothingWeight) 
      : [];
    return getSharedXAxisConfig(xAxisMode, pinnedTime, snaps);
  }, [runIds, runs, metricsDataRef, xAxisMode, smoothingWeight, metricKey]);

  const getSeriesCb = useCallback((pinnedTime: number | null) => {
    if (pinnedTime !== null) {
      const snaps = computePinnedSnapshots(runIds, runs, metricsDataRef, xAxisMode, pinnedTime, [metricKey], smoothingWeight);
      return [
        {
          name: title,
          type: "bar",
          emphasis: { focus: "series" },
          label: { show: false },
          data: snaps.map((s) => {
            const c = runColors[s.id] || "#10b981";
            return { value: Number(s[metricKey]), itemStyle: { color: c }, groupId: s.id };
          }),
        },
      ];
    }

    return runIds.flatMap((id) => {
      const data = metricsDataRef.current[id] || [];
      const run = runs.find((r) => r.id === id);
      const baseTime = run?.start_time ? new Date(run.start_time + "Z").getTime() : Date.now();

      const baseColor = runColors[id] || "#10b981";
      const lineColor = baseColor;

      const pts = data
        .map((d) => {
          let xVal = 0;
          const elapsedSecs = Number(d.elapsed_time || 0);

          if (xAxisMode === "step") xVal = parseInt(d.step, 10) || 0;
          else if (xAxisMode === "absolute") xVal = baseTime + elapsedSecs * 1000;
          else if (xAxisMode === "relative") xVal = elapsedSecs;

          return [xVal, Number(d[metricKey]) || 0];
        })
        .filter((d) => !isNaN(d[1] as number));

      const maxStep = pts.length > 0 ? Math.max(...pts.map((p) => p[0] as number)) : 0;
      const markLineData: any[] = [];

      if (xAxisMode === "step") {
        const startStep = Math.max(50, Math.floor(maxStep / 50) * 50 - 30 * 50);
        const lastMilestone = Math.floor(maxStep / 50) * 50;
        for (let i = startStep; i <= maxStep; i += 50) {
          if (i === 0) continue;
          const isCheckpoint = i % 100 === 0;
          markLineData.push({
            xAxis: i,
            lineStyle: {
              type: isCheckpoint ? "solid" : "dashed",
              color: isCheckpoint ? "rgba(234, 179, 8, 0.4)" : "rgba(168, 85, 247, 0.4)",
              width: 1,
            },
            label: {
              formatter: isCheckpoint ? "Checkpoint" : "Target Sync",
              position: isCheckpoint ? "insideEndTop" : "insideEndBottom",
              color: isCheckpoint ? "rgba(234, 179, 8, 0.7)" : "rgba(168, 85, 247, 0.6)",
              fontSize: 8,
              show: i === lastMilestone,
            },
          });
        }
      }

      let lastEma = pts.length > 0 ? (pts[0][1] as number) : 0;
      const smoothedPts = pts.map((p) => {
        const val = p[1] as number;
        lastEma = lastEma * smoothingWeight + val * (1 - smoothingWeight);
        return [p[0], lastEma];
      });

      const rawSeries = {
        id: `${id}_raw_${metricKey}`,
        name: `Run ${id.substring(0, 4)} (Raw)`,
        type: "line",
        data: pts,
        showSymbol: false,
        symbol: "none",
        smooth: false,
        itemStyle: { color: `${lineColor}` },
        lineStyle: { width: 1, color: `${lineColor}40` },
        tooltip: { show: false },
        groupId: id,
      };

      const smoothedSeries = {
        id: `${id}_smooth_${metricKey}`,
        name: `Run ${id.substring(0, 4)}`,
        type: "line",
        data: smoothedPts,
        showSymbol: false,
        symbol: "circle",
        symbolSize: 4,
        smooth: 0.5,
        itemStyle: { color: lineColor, borderColor: "#000", borderWidth: 1 },
        lineStyle: { width: 1, color: lineColor },
        emphasis: { focus: "series" },
        markLine: { symbol: "none", data: markLineData, animation: false },
        groupId: id,
      };

      return [rawSeries, smoothedSeries];
    });
  }, [runIds, runs, metricsDataRef, runColors, xAxisMode, smoothingWeight, title, metricKey]);

  const { chartRef, chartEvents } = useChartSync(runIds, metricsDataRef, getSeriesCb, getXAxisConfigCb);

  const initialOptions = {
    title: { show: false },
    tooltip: {
      show: false,
      trigger: "axis",
      showDelay: 40,
      transitionDuration: 0,
      backgroundColor: "rgba(9, 9, 11, 0.95)",
      borderColor: "rgba(39, 39, 42, 0.8)",
      borderWidth: 1,
      padding: [4, 8],
      enterable: true,
      extraCssText: "max-height: 300px; overflow-y: auto;",
      textStyle: { color: "#e4e4e7", fontSize: 10, fontWeight: 500 },
      axisPointer: { type: "cross", label: { backgroundColor: "#27272a", fontSize: 10 } },
    },
    grid: { left: "2%", right: "3%", bottom: "5%", top: "15%", containLabel: true },
    xAxis: getXAxisConfigCb(null),
    yAxis: { type: "value" as const, splitLine: { lineStyle: { color: "#27272a" } }, axisLabel: { fontSize: 9 }, scale: true },
    series: [],
    backgroundColor: "transparent",
  };

  return (
    <div className="bg-background flex flex-col relative w-full h-full overflow-hidden p-1 border rounded-md border-border/20">
      <div className="flex items-center justify-center gap-1 z-10 absolute top-2 left-0 right-0 pointer-events-none">
        <span className="text-[10px] font-semibold text-zinc-400 tracking-wider bg-background px-1">
          {title}
        </span>
        <TooltipProvider delayDuration={100}>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="pointer-events-auto cursor-help">
                <Info className="h-3 w-3 text-zinc-500 hover:text-zinc-300 transition-colors" />
              </div>
            </TooltipTrigger>
            <TooltipContent side="top" className="max-w-[200px] text-xs leading-relaxed text-center z-50">
              <p>{description}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
      <ReactECharts
        ref={chartRef}
        option={initialOptions}
        style={{ width: "100%", height: "100%" }}
        className="flex-1 w-full min-h-0"
        theme="dark"
        onEvents={chartEvents}
      />
    </div>
  );
}
