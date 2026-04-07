import { useEffect, useRef } from "react";
import ReactECharts from "echarts-for-react";
import { Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { Run } from "@/bindings/Run";

interface LossStackedAreaProps {
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
}: LossStackedAreaProps) {
  void runColors;
  const chartRef = useRef<ReactECharts>(null);

  const getSeries = () => {
    if (runIds.length === 0) return [];

    const activeRunId =
      runIds.find((id) => metricsDataRef.current[id]?.length > 0) || runIds[0];
    const data = metricsDataRef.current[activeRunId] || [];

    return [
      {
        name: "Policy Loss",
        type: "line",
        areaStyle: { opacity: 0.2 },
        emphasis: { focus: "series" },
        data: data.map((d) => [d.step || 0, d.policy_loss || 0]),
        itemStyle: { color: "#3b82f6" }, // Blue
      },
      {
        name: "Value Loss",
        type: "line",
        areaStyle: { opacity: 0.2 },
        emphasis: { focus: "series" },
        data: data.map((d) => [d.step || 0, d.value_loss || 0]),
        itemStyle: { color: "#8b5cf6" }, // Purple
      },
      {
        name: "Reward Loss",
        type: "line",
        areaStyle: { opacity: 0.2 },
        emphasis: { focus: "series" },
        data: data.map((d) => [d.step || 0, d.reward_loss || 0]),
        itemStyle: { color: "#f59e0b" }, // Amber
      },
    ];
  };

  useEffect(() => {
    let timeoutId: NodeJS.Timeout;
    let isCancelled = false;
    let lastDataLength = -1;

    const renderLoop = () => {
      if (isCancelled) return;

      const currentLength = runIds.reduce(
        (sum, id) => sum + (metricsDataRef.current[id]?.length || 0),
        0,
      );

      if (currentLength !== lastDataLength && chartRef.current) {
        lastDataLength = currentLength;
        const instance = chartRef.current.getEchartsInstance();
        if (instance && !instance.isDisposed()) {
          instance.group = "metricsGroup";
          instance.setOption(
            {
              xAxis: {
                type: "value",
                min: "dataMin",
                max: "dataMax",
              },
              yAxis: {
                type: "value",
              },
              series: getSeries(),
            },
            { replaceMerge: ["xAxis", "yAxis"] },
          );
        }
      }

      timeoutId = setTimeout(renderLoop, 500);
    };

    renderLoop();

    return () => {
      isCancelled = true;
      clearTimeout(timeoutId);
    };
  }, [runIds, runs]);

  const initialOptions = {
    backgroundColor: "transparent",
    tooltip: {
      trigger: "axis",
      axisPointer: {
        type: "cross",
        label: { backgroundColor: "#6a7985" },
      },
    },
    legend: {
      data: ["Policy Loss", "Value Loss", "Reward Loss"],
      top: 25,
      textStyle: { color: "#a1a1aa", fontSize: 10 },
    },
    grid: {
      left: "3%",
      right: "4%",
      bottom: "3%",
      top: "25%",
      containLabel: true,
    },
    xAxis: {
      type: "value",
      boundaryGap: false,
      splitLine: { show: false },
      axisLabel: { fontSize: 9 },
      min: "dataMin",
      max: "dataMax",
    },
    yAxis: {
      type: "value",
      splitLine: { lineStyle: { color: "#27272a" } },
      axisLabel: { fontSize: 9 },
    },
    series: [],
  };

  return (
    <div className="bg-background flex flex-col relative w-full h-full overflow-hidden p-1 border rounded-md border-border/20 min-h-[300px]">
      <div className="absolute top-12 left-2 text-[10px] text-red-500 font-mono z-50 pointer-events-none w-full bg-black/80">
        {(() => {
          if (runIds.length === 0) return "NO DEBUG DATA";
          const activeRunId =
            runIds.find((id) => metricsDataRef.current[id]?.length > 0) ||
            runIds[0];
          const data = metricsDataRef.current[activeRunId];
          if (!data || data.length === 0) return "NO DEBUG DATA";
          const last = data[data.length - 1];
          return `DEBUG DUMP: total=${last.total_loss}, pol=${last.policy_loss}, val=${last.value_loss}, rew=${last.reward_loss}, heat=${last.spatial_heatmap ? last.spatial_heatmap.length : "NULL"}`;
        })()}
      </div>
      <div className="flex items-center justify-between z-10 absolute top-2 left-2 right-2 pointer-events-none">
        <span className="text-[10px] uppercase font-semibold text-zinc-400 tracking-wider bg-background px-1">
          Loss Composition
        </span>
        <TooltipProvider delayDuration={100}>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="pointer-events-auto cursor-help">
                <Info className="h-3 w-3 text-zinc-500 hover:text-zinc-300 transition-colors" />
              </div>
            </TooltipTrigger>
            <TooltipContent
              side="left"
              className="max-w-[200px] text-xs leading-relaxed z-50"
            >
              <p>
                Stacked area graph showing the distinct components of the
                composite loss function during gradient descent.
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
      <ReactECharts
        ref={chartRef}
        option={initialOptions}
        style={{ width: "100%", height: "100%" }}
        theme="dark"
      />
    </div>
  );
}
