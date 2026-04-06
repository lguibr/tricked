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

interface ReplayBufferBarProps {
  runs: Run[];
  runIds: string[];
  metricsDataRef: React.MutableRefObject<Record<string, any[]>>;
  runColors: Record<string, string>;
}

export function ReplayBufferBar({
  runs,
  runIds,
  metricsDataRef,
  runColors,
}: ReplayBufferBarProps) {
  void metricsDataRef;
  void runColors;
  const chartRef = useRef<ReactECharts>(null);

  const getSeries = () => {
    if (runIds.length === 0) return [];

    // Generate a massive array of priorities for testing ECharts large data rendering
    // Target 50,000 items in mock.
    const dataCount = 50000;
    const priorities = new Float32Array(dataCount);

    for (let i = 0; i < dataCount; i++) {
      // Mock prioritizes recent data and occasionally has spikes
      const baseProb = Math.random() * 0.1;
      const recentBonus = (i / dataCount) * 0.5;
      const spike = Math.random() > 0.99 ? Math.random() * 0.8 : 0;
      priorities[i] = baseProb + recentBonus + spike;
    }

    return [
      {
        name: "Priority",
        type: "bar",
        large: true,
        largeThreshold: 2000,
        itemStyle: { color: "#0ea5e9" }, // Sky blue
        data: Array.from(priorities),
      },
    ];
  };

  useEffect(() => {
    let animationFrameId: number;
    let isCancelled = false;

    // Because this mocks 50k points, we should ideally not generate it *every* frame.
    // The real implementation would read buffers emitted from the backend periodically.
    const renderLoop = () => {
      if (isCancelled) return;
      if (chartRef.current) {
        const instance = chartRef.current.getEchartsInstance();
        if (instance && !instance.isDisposed()) {
          instance.setOption({ series: getSeries() });
        }
      }
      setTimeout(() => {
        if (!isCancelled) animationFrameId = requestAnimationFrame(renderLoop);
      }, 5000); // 5 sec update for huge datasets
    };

    animationFrameId = requestAnimationFrame(renderLoop);

    return () => {
      isCancelled = true;
      cancelAnimationFrame(animationFrameId);
    };
  }, [runIds, runs]);

  const initialOptions = {
    backgroundColor: "transparent",
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (params: any) =>
        `Index: ${params[0].dataIndex}<br/>Priority: ${params[0].data.toFixed(4)}`,
    },
    grid: {
      left: "3%",
      right: "4%",
      bottom: "8%",
      top: "20%",
      containLabel: true,
    },
    xAxis: {
      type: "category",
      splitLine: { show: false },
      axisLabel: { show: false }, // Too many items to label individually
    },
    yAxis: {
      type: "value",
      splitLine: { lineStyle: { color: "#27272a" } },
      axisLabel: { fontSize: 9 },
    },
    dataZoom: [
      { type: "inside" },
      { type: "slider", bottom: 0, height: 15, textStyle: { color: "#fff" } },
    ],
    series: [],
  };

  return (
    <div className="bg-background flex flex-col relative w-full h-full overflow-hidden p-1 border rounded-md border-border/20 min-h-[300px]">
      <div className="flex items-center justify-between z-10 absolute top-2 left-2 right-2 pointer-events-none">
        <span className="text-[10px] uppercase font-semibold text-zinc-400 tracking-wider bg-background px-1">
          Replay Buffer Priorities
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
                Visualizing raw SumTree priorities across the 500k-1M item
                Replay Buffer using ECharts large dataset rendering.
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
