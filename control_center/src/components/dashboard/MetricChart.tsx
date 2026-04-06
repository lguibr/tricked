import React, { useEffect, useRef } from "react";
import ReactECharts from "echarts-for-react";
import { Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { Run } from "@/bindings/Run";

interface MetricChartProps {
  title: string;
  description: string;
  metricKey: string;
  runs: Run[];
  runIds: string[];
  metricsDataRef: React.MutableRefObject<Record<string, any[]>>;
  runColors: Record<string, string>;
  xAxisMode?: "step" | "relative" | "absolute";
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
}: MetricChartProps) {
  const chartRef = useRef<ReactECharts>(null);

  const getXAxisConfig = () => {
    if (xAxisMode === "absolute") {
      return {
        type: "time",
        splitLine: { show: false },
        axisLabel: { fontSize: 9 },
        min: "dataMin",
        max: "dataMax",
      };
    } else if (xAxisMode === "relative") {
      return {
        type: "value",
        splitLine: { show: false },
        axisLabel: {
          fontSize: 9,
          formatter: "{value} s",
        },
        min: "dataMin",
        max: "dataMax",
      };
    }
    return {
      type: "value",
      splitLine: { show: false },
      axisLabel: { fontSize: 9 },
      min: "dataMin",
      max: "dataMax",
    };
  };

  const getSeries = () => {
    return runIds.map((id) => {
      const data = metricsDataRef.current[id] || [];
      const run = runs.find((r) => r.id === id);
      const baseTime = run?.start_time
        ? new Date(run.start_time + "Z").getTime()
        : Date.now();

      return {
        name: `Run ${id.substring(0, 4)}`,
        type: "line",
        showSymbol: false,
        smooth: true,
        itemStyle: { color: runColors[id] || "#10b981" },
        data: data
          .map((d) => {
            let xVal = 0;
            const elapsedSecs = Number(d.elapsed_time || 0);

            if (xAxisMode === "step") {
              xVal = parseInt(d.step, 10) || 0;
            } else if (xAxisMode === "absolute") {
              xVal = baseTime + elapsedSecs * 1000;
            } else if (xAxisMode === "relative") {
              xVal = elapsedSecs; // seconds
            }

            return [xVal, Number(d[metricKey]) || 0];
          })
          .filter((d) => !isNaN(d[1])),
      };
    });
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
      animationFrameId = requestAnimationFrame(renderLoop);
    };

    animationFrameId = requestAnimationFrame(renderLoop);

    return () => {
      isCancelled = true;
      cancelAnimationFrame(animationFrameId);
    };
  }, [runIds, runs, xAxisMode, runColors]);

  const initialOptions = {
    title: { show: false },
    tooltip: { trigger: "axis" as const },
    grid: {
      left: "2%",
      right: "3%",
      bottom: "5%",
      top: "15%",
      containLabel: true,
    },
    xAxis: getXAxisConfig(),
    yAxis: {
      type: "value" as const,
      splitLine: { lineStyle: { color: "#27272a" } },
      axisLabel: { fontSize: 9 },
      scale: true,
    },
    series: [],
    backgroundColor: "transparent",
  };

  return (
    <div className="bg-background flex flex-col relative w-full h-full overflow-hidden p-1 border rounded-md border-border/20">
      <div className="flex items-center justify-center gap-1 z-10 absolute top-2 left-0 right-0 pointer-events-none">
        <span className="text-[10px] uppercase font-semibold text-zinc-400 tracking-wider bg-background px-1">
          {title}
        </span>
        <TooltipProvider delayDuration={100}>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="pointer-events-auto cursor-help">
                <Info className="h-3 w-3 text-zinc-500 hover:text-zinc-300 transition-colors" />
              </div>
            </TooltipTrigger>
            <TooltipContent
              side="top"
              className="max-w-[200px] text-xs leading-relaxed text-center z-50"
            >
              <p>{description}</p>
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
