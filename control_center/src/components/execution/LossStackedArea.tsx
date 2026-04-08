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

  const hexToHSL = (H: string): [number, number, number] => {
    let r = 0, g = 0, b = 0;
    if (H.length === 4) {
      r = parseInt(H[1] + H[1], 16);
      g = parseInt(H[2] + H[2], 16);
      b = parseInt(H[3] + H[3], 16);
    } else if (H.length === 7) {
      r = parseInt(H.slice(1, 3), 16);
      g = parseInt(H.slice(3, 5), 16);
      b = parseInt(H.slice(5, 7), 16);
    }
    r /= 255; g /= 255; b /= 255;
    const cmin = Math.min(r, g, b), cmax = Math.max(r, g, b), delta = cmax - cmin;
    let h = 0, s = 0, l = 0;
    if (delta === 0) h = 0;
    else if (cmax === r) h = ((g - b) / delta) % 6;
    else if (cmax === g) h = (b - r) / delta + 2;
    else h = (r - g) / delta + 4;
    h = Math.round(h * 60);
    if (h < 0) h += 360;
    l = (cmax + cmin) / 2;
    s = delta === 0 ? 0 : delta / (1 - Math.abs(2 * l - 1));
    return [h, +(s * 100).toFixed(1), +(l * 100).toFixed(1)];
  };

  const getSeries = () => {
    if (runIds.length === 0) return [];

    const activeRunId =
      runIds.find((id) => metricsDataRef.current[id]?.length > 0) || runIds[0];
    const data = metricsDataRef.current[activeRunId] || [];

    const baseColor = runColors[activeRunId] || "#3b82f6";
    const [h, s, l] = hexToHSL(baseColor);

    // Create 3 different visual shades by interpolating lightness
    const l1 = Math.max(10, l - 20); // Darker
    const l2 = l;                    // Base
    const l3 = Math.min(90, l + 20); // Lighter

    return [
      {
        id: "policy_loss",
        name: "Policy Loss",
        type: "line",
        areaStyle: { opacity: 0.2 },
        emphasis: { focus: "series" },
        data: data.map((d) => [d.step || 0, d.policy_loss || 0]),
        itemStyle: { color: `hsl(${h}, ${s}%, ${l1}%)` },
      },
      {
        id: "value_loss",
        name: "Value Loss",
        type: "line",
        areaStyle: { opacity: 0.2 },
        emphasis: { focus: "series" },
        data: data.map((d) => [d.step || 0, d.value_loss || 0]),
        itemStyle: { color: `hsl(${h}, ${s}%, ${l2}%)` },
      },
      {
        id: "reward_loss",
        name: "Reward Loss",
        type: "line",
        areaStyle: { opacity: 0.2 },
        emphasis: { focus: "series" },
        data: data.map((d) => [d.step || 0, d.reward_loss || 0]),
        itemStyle: { color: `hsl(${h}, ${s}%, ${l3}%)` },
      },
    ];
  };

  useEffect(() => {
    let timeoutId: NodeJS.Timeout;
    let isCancelled = false;
    let lastDataLength = -1;
    let lastRunIds: string[] = [];

    const renderLoop = () => {
      if (isCancelled) return;

      const currentLength = runIds.reduce(
        (sum, id) => sum + (metricsDataRef.current[id]?.length || 0),
        0,
      );

      const runIdsChanged = runIds.join(",") !== lastRunIds.join(",");

      if ((currentLength !== lastDataLength || runIdsChanged) && chartRef.current) {
        lastDataLength = currentLength;
        lastRunIds = [...runIds];
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
            { replaceMerge: ["xAxis", "yAxis", "series"] },
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
