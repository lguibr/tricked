import React, { useEffect, useRef } from "react";
import * as echarts from "echarts";
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
  metricIndex?: number;
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
  metricIndex = 0,
  smoothingWeight = 0.9,
}: MetricChartProps) {
  const chartRef = useRef<ReactECharts>(null);

  const hexToHSL = (H: string): [number, number, number] => {
    let r = 0,
      g = 0,
      b = 0;
    if (H.length === 4) {
      r = parseInt(H[1] + H[1], 16);
      g = parseInt(H[2] + H[2], 16);
      b = parseInt(H[3] + H[3], 16);
    } else if (H.length === 7) {
      r = parseInt(H.slice(1, 3), 16);
      g = parseInt(H.slice(3, 5), 16);
      b = parseInt(H.slice(5, 7), 16);
    }
    r /= 255;
    g /= 255;
    b /= 255;
    const cmin = Math.min(r, g, b),
      cmax = Math.max(r, g, b),
      delta = cmax - cmin;
    let h = 0,
      s = 0,
      l = 0;
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

  const getXAxisConfig = () => {
    if (xAxisMode === "absolute") {
      return {
        type: "time",
        boundaryGap: false,
        splitLine: { show: false },
        axisLabel: { fontSize: 9 },
        min: "dataMin",
        max: "dataMax",
      };
    } else if (xAxisMode === "relative") {
      return {
        type: "value",
        boundaryGap: false,
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
      boundaryGap: false,
      splitLine: { show: false },
      axisLabel: { fontSize: 9 },
      min: "dataMin",
      max: "dataMax",
    };
  };

  const getSeries = () => {
    return runIds.flatMap((id) => {
      const data = metricsDataRef.current[id] || [];
      const run = runs.find((r) => r.id === id);
      const baseTime = run?.start_time
        ? new Date(run.start_time + "Z").getTime()
        : Date.now();

      const baseColor = runColors[id] || "#10b981";
      const [bh, bs, bl] = hexToHSL(baseColor);
      const shiftedHue = (bh + metricIndex * 25) % 360;
      const lineColor = `hsl(${shiftedHue}, ${bs}%, ${bl}%)`;

      const pts = data
        .map((d) => {
          let xVal = 0;
          const elapsedSecs = Number(d.elapsed_time || 0);

          if (xAxisMode === "step") {
            xVal = parseInt(d.step, 10) || 0;
          } else if (xAxisMode === "absolute") {
            xVal = baseTime + elapsedSecs * 1000;
          } else if (xAxisMode === "relative") {
            xVal = elapsedSecs;
          }

          return [xVal, Number(d[metricKey]) || 0];
        })
        .filter((d) => !isNaN(d[1]));

      const maxStep =
        pts.length > 0 ? Math.max(...pts.map((p) => p[0] as number)) : 0;
      const markLineData: any[] = [];

      if (xAxisMode === "step") {
        // Only show last 30 milestone markers to prevent visual clutter
        const startStep = Math.max(50, Math.floor(maxStep / 50) * 50 - 30 * 50);
        const lastMilestone = Math.floor(maxStep / 50) * 50;
        for (let i = startStep; i <= maxStep; i += 50) {
          if (i === 0) continue;
          const isCheckpoint = i % 100 === 0;
          markLineData.push({
            xAxis: i,
            lineStyle: {
              type: isCheckpoint ? "solid" : "dashed",
              color: isCheckpoint
                ? "rgba(234, 179, 8, 0.4)"
                : "rgba(168, 85, 247, 0.4)",
              width: 1,
            },
            label: {
              formatter: isCheckpoint ? "Checkpoint" : "Target Sync",
              position: isCheckpoint ? "insideEndTop" : "insideEndBottom",
              color: isCheckpoint
                ? "rgba(234, 179, 8, 0.7)"
                : "rgba(168, 85, 247, 0.6)",
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
        name: `Run ${id.substring(0, 4)} (Raw)`,
        type: "line",
        data: pts,
        showSymbol: false,
        symbol: "none",
        smooth: false,
        itemStyle: { color: `${lineColor}` },
        lineStyle: { width: 1, color: `${lineColor}40` },
        tooltip: { show: false },
      };

      const smoothedSeries = {
        name: `Run ${id.substring(0, 4)}`,
        type: "line",
        data: smoothedPts,
        showSymbol: false,
        symbol: "circle",
        symbolSize: 4,
        smooth: true,
        itemStyle: { color: lineColor, borderColor: "#000", borderWidth: 1 },
        lineStyle: { width: 2, color: lineColor },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: `${baseColor}20` },
            { offset: 1, color: `${baseColor}00` },
          ]),
        },
        emphasis: {
          focus: "series",
        },
        markLine: {
          symbol: "none",
          data: markLineData,
          animation: false,
        },
      };

      return [rawSeries, smoothedSeries];
    });
  };

  useEffect(() => {
    let timeoutId: NodeJS.Timeout;
    let isCancelled = false;
    let lastDataLength = -1;
    let lastSmoothingWeight = -1;
    let lastAxisMode = "";

    const renderLoop = () => {
      if (isCancelled) return;

      const currentLength = runIds.reduce((sum: number, id: string) => {
        return sum + (metricsDataRef.current[id]?.length || 0);
      }, 0);

      const needsRender =
        currentLength !== lastDataLength ||
        lastSmoothingWeight !== smoothingWeight ||
        lastAxisMode !== xAxisMode;

      if (needsRender && chartRef.current) {
        const instance = chartRef.current.getEchartsInstance();
        if (instance && !instance.isDisposed()) {
          const dom = instance.getDom();
          if (dom && dom.clientWidth > 0 && dom.clientHeight > 0) {
            lastDataLength = currentLength;
            lastSmoothingWeight = smoothingWeight;
            lastAxisMode = xAxisMode;

            instance.group = "metricsGroup";
            instance.setOption({
              xAxis: getXAxisConfig(),
              yAxis: { type: "value" },
              series: getSeries(),
            });
          }
        }
      }
      timeoutId = setTimeout(renderLoop, 500);
    };

    renderLoop();

    return () => {
      isCancelled = true;
      clearTimeout(timeoutId);
    };
  }, [runIds, runs, xAxisMode, runColors, metricIndex]);

  const initialOptions = {
    title: { show: false },
    tooltip: {
      trigger: "axis",
      backgroundColor: "rgba(9, 9, 11, 0.95)",
      borderColor: "rgba(39, 39, 42, 0.8)",
      borderWidth: 1,
      padding: [4, 8],
      textStyle: {
        color: "#e4e4e7",
        fontSize: 10,
        fontWeight: 500,
      },
      axisPointer: {
        type: "cross",
        label: {
          backgroundColor: "#27272a",
          fontSize: 10,
        },
      },
    },
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
        className="flex-1 w-full min-h-0"
        theme="dark"
      />
    </div>
  );
}
