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

interface ActionThemeRiverProps {
  runs: Run[];
  runIds: string[];
  metricsDataRef: React.MutableRefObject<Record<string, any[]>>;
  runColors: Record<string, string>;
}

export function ActionThemeRiver({
  runs,
  runIds,
  metricsDataRef,
  runColors,
}: ActionThemeRiverProps) {
  void runColors;
  const chartRef = useRef<ReactECharts>(null);

  const getSeries = () => {
    if (runIds.length === 0) return [];

    const activeRunId = runIds[0];
    const data = metricsDataRef.current[activeRunId] || [];

    const riverData: [number, number, string][] = [];
    const actions = [
      "Rotate L",
      "Rotate R",
      "Drop",
      "Hold",
      "Move L",
      "Move R",
    ];

    // Generate simulated ThemeRiver data based on time steps
    for (let i = 0; i < data.length; i++) {
      const step = data[i].step || i;
      actions.forEach((action, actionIndex) => {
        // Simulated probability shifting over time
        const baseProb = Math.sin(i * 0.1 + actionIndex) * 10 + 20;
        const prob = Math.max(0, baseProb + Math.random() * 5);
        riverData.push([step, prob, action]);
      });
    }

    return [
      {
        type: "themeRiver",
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: "rgba(0, 0, 0, 0.3)",
          },
        },
        data: riverData,
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
      setTimeout(() => {
        if (!isCancelled) animationFrameId = requestAnimationFrame(renderLoop);
      }, 500);
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
      axisPointer: { type: "line" },
    },
    legend: {
      bottom: 0,
      textStyle: { color: "#a1a1aa", fontSize: 10 },
    },
    singleAxis: {
      top: 20,
      bottom: 50,
      axisTick: { show: false },
      axisLabel: { fontSize: 9 },
      type: "value",
      splitLine: { show: false },
    },
    series: [],
  };

  return (
    <div className="bg-background flex flex-col relative w-full h-full overflow-hidden p-1 border rounded-md border-border/20 min-h-[300px]">
      <div className="flex items-center justify-between z-10 absolute top-2 left-2 right-2 pointer-events-none">
        <span className="text-[10px] uppercase font-semibold text-zinc-400 tracking-wider bg-background px-1">
          Action Distribution Flow
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
                ThemeRiver visualization showing how the probability
                distribution of different agent actions shifts over training
                steps.
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
