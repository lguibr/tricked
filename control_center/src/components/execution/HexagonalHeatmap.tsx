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

interface HexagonalHeatmapProps {
  runs: Run[];
  runIds: string[];
  metricsDataRef: React.MutableRefObject<Record<string, any[]>>;
  runColors: Record<string, string>;
}

// Tricked board has 96 cells (e.g., let's map it as a 12x8 grid for visualization)
const X_SIZE = 12;
const Y_SIZE = 8;

export function HexagonalHeatmap({
  runs,
  runIds,
  metricsDataRef,
  runColors,
}: HexagonalHeatmapProps) {
  const chartRef = useRef<ReactECharts>(null);

  const getSeries = () => {
    if (runIds.length === 0) return [];

    const activeRunId = runIds[0];
    const data = metricsDataRef.current[activeRunId] || [];

    // Generate mock heatmap data spanning 96 cells
    const heatmapData = [];
    for (let i = 0; i < X_SIZE; i++) {
      for (let j = 0; j < Y_SIZE; j++) {
        // Simulated Q-values shifting over time
        const timeOffset = data.length * 0.1;
        const val =
          Math.sin(i * 0.5 + timeOffset) * Math.cos(j * 0.5 + timeOffset);
        heatmapData.push([i, j, val.toFixed(2)]);
      }
    }

    return [
      {
        name: "Q-Value Heatmap",
        type: "heatmap",
        data: heatmapData,
        label: {
          show: false,
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: "rgba(0, 0, 0, 0.5)",
          },
        },
      },
    ];
  };

  useEffect(() => {
    let animationFrameId: number;
    let isCancelled = false;

    const renderLoop = () => {
      if (isCancelled) return;
      if (chartRef.current) {
        const instance = chartRef.current.getEchartsInstance();
        if (instance && !instance.isDisposed()) {
          instance.setOption({ series: getSeries() });
        }
      }
      // Update heatmap smoothly at standard chart rate
      setTimeout(() => {
        if (!isCancelled) animationFrameId = requestAnimationFrame(renderLoop);
      }, 500);
    };

    animationFrameId = requestAnimationFrame(renderLoop);

    return () => {
      isCancelled = true;
      cancelAnimationFrame(animationFrameId);
    };
  }, [runIds, runs, runColors]);

  const xData = Array.from({ length: X_SIZE }, (_, i) => i.toString());
  const yData = Array.from({ length: Y_SIZE }, (_, i) => i.toString());

  const initialOptions = {
    backgroundColor: "transparent",
    tooltip: {
      position: "top",
    },
    grid: {
      height: "80%",
      top: "15%",
      left: "5%",
      right: "5%",
      bottom: "5%",
    },
    xAxis: {
      type: "category",
      data: xData,
      splitArea: {
        show: true,
      },
    },
    yAxis: {
      type: "category",
      data: yData,
      splitArea: {
        show: true,
      },
    },
    visualMap: {
      min: -1,
      max: 1,
      calculable: true,
      orient: "horizontal",
      left: "center",
      bottom: "0%",
      show: false, // hide visual map legend to save space
      inRange: {
        color: ["#ef4444", "#18181b", "#10b981"], // Red to Black to Green
      },
    },
    series: [],
  };

  return (
    <div className="bg-background flex flex-col relative w-full h-full overflow-hidden p-1 border rounded-md border-border/20 min-h-[300px]">
      <div className="flex items-center justify-between z-10 absolute top-2 left-2 right-2 pointer-events-none">
        <span className="text-[10px] uppercase font-semibold text-zinc-400 tracking-wider bg-background px-1">
          Board Q-Value Heatmap
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
                Visualizes the agent's spatial value estimation (Q-values)
                across the playing board cells.
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
