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
import { hexToHSL, computePinnedSnapshots, getSharedXAxisConfig } from "@/lib/chart-utils";
import { useChartSync } from "@/hooks/useChartSync";

interface LossStackedAreaProps {
  xAxisMode: "step" | "relative" | "absolute";
  smoothingWeight: number;
  runs: Run[];
  runIds: string[];
  metricsDataRef: React.MutableRefObject<Record<string, any[]>>;
  runColors: Record<string, string>;
}

export function LossStackedArea({
  runs,
  runIds,
  metricsDataRef,
  runColors,
  xAxisMode,
  smoothingWeight,
}: LossStackedAreaProps) {
  
  const getXAxisConfigCb = useCallback((pinnedTime: number | null) => {
    const snaps = pinnedTime !== null 
      ? computePinnedSnapshots(runIds, runs, metricsDataRef, xAxisMode, pinnedTime, ["policy_loss", "value_loss", "reward_loss"], smoothingWeight) 
      : [];
    return getSharedXAxisConfig(xAxisMode, pinnedTime, snaps);
  }, [runIds, runs, metricsDataRef, xAxisMode, smoothingWeight]);

  const getSeriesCb = useCallback((pinnedTime: number | null) => {
    if (pinnedTime !== null) {
      const snaps = computePinnedSnapshots(runIds, runs, metricsDataRef, xAxisMode, pinnedTime, ["policy_loss", "value_loss", "reward_loss"], smoothingWeight);
      return [
        {
          name: "Policy Loss",
          type: "bar",
          stack: "total",
          emphasis: { focus: "series" },
          label: { show: false },
          data: snaps.map((s) => {
            const c = runColors[s.id] || "#3b82f6";
            const [h, st, l] = hexToHSL(c);
            return {
              value: Number(s.policy_loss),
              itemStyle: { color: `hsl(${h}, ${st}%, ${Math.max(10, l - 20)}%)` },
              groupId: s.id,
            };
          }),
        },
        {
          name: "Value Loss",
          type: "bar",
          stack: "total",
          emphasis: { focus: "series" },
          label: { show: false },
          data: snaps.map((s) => {
            const c = runColors[s.id] || "#3b82f6";
            const [h, st, l] = hexToHSL(c);
            return {
              value: Number(s.value_loss),
              itemStyle: { color: `hsl(${h}, ${st}%, ${l}%)` },
              groupId: s.id,
            };
          }),
        },
        {
          name: "Reward Loss",
          type: "bar",
          stack: "total",
          emphasis: { focus: "series" },
          label: { show: false },
          data: snaps.map((s) => {
            const c = runColors[s.id] || "#3b82f6";
            const [h, st, l] = hexToHSL(c);
            return {
              value: Number(s.reward_loss),
              itemStyle: { color: `hsl(${h}, ${st}%, ${Math.min(90, l + 20)}%)` },
              groupId: s.id,
            };
          }),
        },
      ];
    }

    return runIds.flatMap((id) => {
      const rawData = metricsDataRef.current[id] || [];
      const baseColor = runColors[id] || "#3b82f6";
      const [h, s, l] = hexToHSL(baseColor);

      const l1 = Math.max(10, l - 20);
      const l2 = l;
      const l3 = Math.min(90, l + 20);

      let lastPol = rawData.length > 0 ? Number(rawData[0].policy_loss) || 0 : 0;
      let lastVal = rawData.length > 0 ? Number(rawData[0].value_loss) || 0 : 0;
      let lastRew = rawData.length > 0 ? Number(rawData[0].reward_loss) || 0 : 0;

      const run = runs.find((r) => r.id === id);
      const baseTime = run?.start_time ? new Date(run.start_time + "Z").getTime() : Date.now();

      const smoothedData = rawData.map((d) => {
        let xVal = 0;
        const elapsedSecs = Number(d.elapsed_time || 0);

        if (xAxisMode === "step") xVal = parseInt(d.step || 0, 10);
        else if (xAxisMode === "absolute") xVal = baseTime + elapsedSecs * 1000;
        else if (xAxisMode === "relative") xVal = elapsedSecs;

        const curPol = Number(d.policy_loss) || 0;
        const curVal = Number(d.value_loss) || 0;
        const curRew = Number(d.reward_loss) || 0;

        if (!isNaN(curPol)) lastPol = lastPol * smoothingWeight + curPol * (1 - smoothingWeight);
        if (!isNaN(curVal)) lastVal = lastVal * smoothingWeight + curVal * (1 - smoothingWeight);
        if (!isNaN(curRew)) lastRew = lastRew * smoothingWeight + curRew * (1 - smoothingWeight);

        return { xVal, pol: lastPol, val: lastVal, rew: lastRew };
      });

      return [
        {
          id: `${id}_policy_loss`,
          name: `Run ${id.substring(0, 4)} Pol`,
          type: "line",
          smooth: 0.5,
          lineStyle: { width: 1 },
          showSymbol: false,
          symbol: "none",
          emphasis: { focus: "series" },
          data: smoothedData.map((d) => [d.xVal, d.pol]).filter((d) => !isNaN(d[1] as number)),
          itemStyle: { color: `hsl(${h}, ${s}%, ${l1}%)` },
          groupId: id,
        },
        {
          id: `${id}_value_loss`,
          name: `Run ${id.substring(0, 4)} Val`,
          type: "line",
          smooth: 0.5,
          lineStyle: { width: 1 },
          showSymbol: false,
          symbol: "none",
          emphasis: { focus: "series" },
          data: smoothedData.map((d) => [d.xVal, d.val]).filter((d) => !isNaN(d[1] as number)),
          itemStyle: { color: `hsl(${h}, ${s}%, ${l2}%)` },
          groupId: id,
        },
        {
          id: `${id}_reward_loss`,
          name: `Run ${id.substring(0, 4)} Rew`,
          type: "line",
          smooth: 0.5,
          lineStyle: { width: 1 },
          showSymbol: false,
          symbol: "none",
          emphasis: { focus: "series" },
          data: smoothedData.map((d) => [d.xVal, d.rew]).filter((d) => !isNaN(d[1] as number)),
          itemStyle: { color: `hsl(${h}, ${s}%, ${l3}%)` },
          groupId: id,
        },
      ];
    });
  }, [runIds, runs, metricsDataRef, runColors, xAxisMode, smoothingWeight]);

  const { chartRef, chartEvents } = useChartSync(runIds, metricsDataRef, getSeriesCb, getXAxisConfigCb);

  const initialOptions = {
    backgroundColor: "transparent",
    tooltip: {
      show: false,
      trigger: "axis",
      axisPointer: { type: "cross", label: { backgroundColor: "#27272a", fontSize: 10 } },
    },
    legend: { top: 25, textStyle: { color: "#a1a1aa", fontSize: 10 }, type: "scroll" },
    grid: { left: "3%", right: "4%", bottom: "3%", top: "25%", containLabel: true },
    xAxis: getXAxisConfigCb(null),
    yAxis: { type: "value", splitLine: { lineStyle: { color: "#27272a" } }, axisLabel: { fontSize: 9 } },
    series: [],
  };

  return (
    <div className="bg-background flex flex-col relative w-full h-full overflow-hidden p-1 border rounded-md border-border/20 min-h-[300px]">
      <div className="absolute top-12 left-2 text-[10px] text-red-500 font-mono z-50 pointer-events-none w-full bg-black/80">
        {(() => {
          if (runIds.length === 0) return "NO DEBUG DATA";
          const activeRunId = runIds.find((id) => metricsDataRef.current[id]?.length > 0) || runIds[0];
          const data = metricsDataRef.current[activeRunId];
          if (!data || data.length === 0) return "NO DEBUG DATA";
          const last = data[data.length - 1];
          return `DEBUG DUMP: total=${last.total_loss}, pol=${last.policy_loss}, val=${last.value_loss}, rew=${last.reward_loss}, heat=${last.spatial_heatmap ? last.spatial_heatmap.length : "NULL"}`;
        })()}
      </div>
      <div className="flex items-center justify-between z-10 absolute top-2 left-2 right-2 pointer-events-none">
        <span className="text-[10px] font-semibold text-zinc-400 tracking-wider bg-background px-1">
          Loss Composition
        </span>
        <TooltipProvider delayDuration={100}>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="pointer-events-auto cursor-help">
                <Info className="h-3 w-3 text-zinc-500 hover:text-zinc-300 transition-colors" />
              </div>
            </TooltipTrigger>
            <TooltipContent side="left" className="max-w-[200px] text-xs leading-relaxed z-50">
              <p>Stacked area graph showing the distinct components of the composite loss function during gradient descent.</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
      <ReactECharts
        ref={chartRef}
        option={initialOptions}
        style={{ width: "100%", height: "100%" }}
        theme="dark"
        onEvents={chartEvents}
      />
    </div>
  );
}
