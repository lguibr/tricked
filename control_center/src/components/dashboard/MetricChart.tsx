import ReactECharts from "echarts-for-react";
import { Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface MetricChartProps {
  title: string;
  description: string;
  metricKey: string;
  runIds: string[];
  metricsData: Record<string, any[]>;
  runColors: Record<string, string>;
  xAxisMode?: "step" | "relative" | "absolute";
}

export function MetricChart({
  title,
  description,
  metricKey,
  runIds,
  metricsData,
  runColors,
  xAxisMode = "step",
}: MetricChartProps) {
  const series = runIds.map((id) => {
    const data = metricsData[id] || [];
    const baseTime =
      data.length > 0 && data[0].timestamp
        ? new Date(data[0].timestamp).getTime()
        : null;

    return {
      name: `Run ${id.substring(0, 4)}`,
      type: "line",
      showSymbol: false,
      smooth: true,
      itemStyle: { color: runColors[id] || "#10b981" },
      data: data
        .map((d) => {
          let xVal = 0;
          const ts = d.timestamp ? new Date(d.timestamp).getTime() : 0;

          if (xAxisMode === "step") {
            xVal = parseInt(d.step, 10) || 0;
          } else if (xAxisMode === "absolute") {
            xVal = ts;
          } else if (xAxisMode === "relative") {
            xVal = baseTime ? (ts - baseTime) / 1000 : 0; // seconds
          }

          return [xVal, parseFloat(d[metricKey]) || 0];
        })
        .filter((d) => !isNaN(d[1])),
    };
  });

  const getXAxisConfig = () => {
    if (xAxisMode === "absolute") {
      return {
        type: "time",
        splitLine: { show: false },
        axisLabel: { fontSize: 9 },
      };
    } else if (xAxisMode === "relative") {
      return {
        type: "value",
        splitLine: { show: false },
        axisLabel: {
          fontSize: 9,
          formatter: "{value} s",
        },
      };
    }
    return {
      type: "value",
      splitLine: { show: false },
      axisLabel: { fontSize: 9 },
    };
  };

  const options = {
    title: { show: false },
    tooltip: { trigger: "axis" },
    grid: {
      left: "2%",
      right: "3%",
      bottom: "5%",
      top: "15%",
      containLabel: true,
    },
    xAxis: getXAxisConfig(),
    yAxis: {
      type: "value",
      splitLine: { lineStyle: { color: "#27272a" } },
      axisLabel: { fontSize: 9 },
    },
    series,
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
        option={options}
        style={{ width: "100%", height: "100%" }}
        theme="dark"
      />
    </div>
  );
}
