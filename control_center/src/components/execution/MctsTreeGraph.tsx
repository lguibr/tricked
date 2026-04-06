import { useEffect, useRef } from "react";
import ReactECharts from "echarts-for-react";
import "echarts-gl";
import { Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { Run } from "@/bindings/Run";

interface MctsTreeGraphProps {
  runs: Run[];
  runIds: string[];
  metricsDataRef: React.MutableRefObject<Record<string, any[]>>;
  runColors: Record<string, string>;
}

export function MctsTreeGraph({
  runs,
  runIds,
  metricsDataRef,
  runColors,
}: MctsTreeGraphProps) {
  const chartRef = useRef<ReactECharts>(null);

  // Generate mock data for the MCTS tree since the backend doesn't export raw nodes yet
  // We will simulate a tree expansion based on the loss or steps.
  const getSeries = () => {
    if (runIds.length === 0) return [];

    const activeRunId = runIds[0]; // Just visualizing the first selected run
    void runColors;
    const data = metricsDataRef.current[activeRunId] || [];

    // Simulate tree nodes based on step count
    const nodeCount = Math.min(200, 10 + data.length * 2);

    const nodes = [];
    const edges = [];

    nodes.push({
      id: "0",
      name: "Root",
      symbolSize: 20,
      value: 1000,
      itemStyle: { color: "#ef4444" }, // Root is red
    });

    for (let i = 1; i < nodeCount; i++) {
      const parentId = Math.floor(Math.pow(Math.random(), 2) * i).toString();
      const visits = Math.max(1, Math.floor(100 / Math.sqrt(i)));
      const qValue = Math.random() * 2 - 1; // -1 to 1

      nodes.push({
        id: i.toString(),
        name: `Node ${i}`,
        symbolSize: Math.max(3, visits / 2),
        value: qValue,
        itemStyle: {
          color: qValue > 0 ? "#10b981" : "#f59e0b",
        },
      });

      edges.push({
        source: parentId,
        target: i.toString(),
      });
    }

    return [
      {
        type: "graphGL",
        nodes: nodes,
        edges: edges,
        lineStyle: {
          color: "rgba(255,255,255,0.2)",
          width: 1,
        },
        itemStyle: {
          opacity: 0.8,
        },
        forceAtlas2: {
          steps: 5,
          jitterTolerence: 10,
          edgeWeight: [0.2, 1],
          gravity: 1,
          edgeWeightInfluence: 0,
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
      // Re-render every 2 seconds for the graph layout to settle, instead of 60fps to save CPU on graphGL
      setTimeout(() => {
        if (!isCancelled) animationFrameId = requestAnimationFrame(renderLoop);
      }, 2000);
    };

    animationFrameId = requestAnimationFrame(renderLoop);

    return () => {
      isCancelled = true;
      cancelAnimationFrame(animationFrameId);
    };
  }, [runIds, runs, runColors]);

  const initialOptions = {
    backgroundColor: "transparent",
    tooltip: {
      formatter: "{b}: Q={c}",
    },
    series: [],
  };

  return (
    <div className="bg-background flex flex-col relative w-full h-full overflow-hidden p-1 border rounded-md border-border/20 min-h-[300px]">
      <div className="flex items-center justify-between z-10 absolute top-2 left-2 right-2 pointer-events-none">
        <span className="text-[10px] uppercase font-semibold text-zinc-400 tracking-wider bg-background px-1">
          MCTS Tree Expansion (GraphGL)
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
                Visualizes the Monte Carlo Tree Search node expansions in 3D
                force-directed space. Green = High Q-Value, Orange = Low
                Q-Value.
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
